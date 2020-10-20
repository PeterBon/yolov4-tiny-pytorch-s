# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataset import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo4_tiny import YoloBody
from tqdm import tqdm
import yaml
import math
from tensorboardX import SummaryWriter


def init_weights(model):
    # 进行权值初始化
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(2):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)
            # train_writer.add_scalar('loss_batch', loss, (epoch * epoch_size + iteration))

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(2):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss

                # val_writer.add_scalar('loss_batch', loss, (epoch * epoch_size_val + iteration))

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    # tensorboardX
    writer.add_scalars('loss', {'train': total_loss / (epoch_size + 1), 'val': val_loss / (epoch_size_val + 1)}, epoch)
    writer.add_scalar('lr', get_lr(optimizer), epoch)
    writer.flush()

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    # log.yaml
    avg_train_loss = total_loss / (epoch_size + 1)
    avg_train_loss = avg_train_loss.item()
    avg_val_loss = val_loss / (epoch_size_val + 1)
    avg_val_loss = avg_val_loss.item()
    log['epoch_number'] += 1
    log['Epoch%03d' % (epoch + 1)] = [avg_train_loss, avg_val_loss]
    if log['best_val_loss'] < 0 or avg_val_loss < log['best_val_loss']:
        log['best_val_loss'] = avg_val_loss
        torch.save(model.state_dict(), 'logs/best.pth')
    with open('logs/log.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(log, f)

    torch.save(model.state_dict(), 'logs/last.pth')


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # hyp
    with open('model_data/hyp.yaml', encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # log
    if os.path.exists('logs/log.yaml'):
        with open('logs/log.yaml', encoding='utf-8') as f:
            log = yaml.load(f, Loader=yaml.FullLoader)
    else:
        log = {'epoch_number': 0, 'best_val_loss': -1}

    #   输入的shape大小
    input_shape = hyp.get('input_shape')

    class_names = get_classes(hyp.get('classes_path'))
    anchors = get_anchors(hyp.get('anchors_path'))
    num_classes = len(class_names)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)

    model_path = hyp.get('model_path')

    if model_path:
        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished!')
    else:
        init_weights(model)

    net = model.train()

    Cuda = torch.cuda.is_available()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(2):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, \
                                    (input_shape[1], input_shape[0]), hyp.get('smoooth_label'), Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = hyp.get('val_split')
    with open(hyp.get('annotation_path')) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # tensorboardX
    writer = SummaryWriter(logdir='logs')
    if Cuda:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[0], input_shape[1])).type(
            torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[0], input_shape[1])).type(torch.FloatTensor)
    writer.add_graph(model, (graph_inputs,))

    Batch_size = hyp.get('batch_size')
    start_epoch = hyp.get('start_epoch')
    end_epoch = hyp.get('end_epoch')
    optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': hyp.get('lr')}], lr=hyp.get('lr'),
                           weight_decay=hyp.get('weight_decay'))
    if hyp.get('lr_scheduler') == 'cosine':
        lf = lambda x: ((1 + math.cos(x * math.pi / hyp.get('epochs'))) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5,last_epoch=start_epoch - 1)
    else:
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=start_epoch - 1)
        func = lambda epoch: hyp.get('gamma') ** epoch
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func, last_epoch=start_epoch - 1)

    train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), hyp=hyp)
    val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), hyp=hyp)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_size = max(1, num_train // Batch_size)
    epoch_size_val = num_val // Batch_size
    if hyp.get('freeze'):
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, end_epoch, Cuda)
        lr_scheduler.step()

    writer.close()
