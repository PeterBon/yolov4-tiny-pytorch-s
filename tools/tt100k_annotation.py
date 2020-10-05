import json
import os

datadir = "C:/Users/Peter/PycharmProjects/datasets/tt100k"  # 数据目录
filedir = datadir + "/annotations.json"  # 标签文件
ids_train = open(datadir + "/train/ids.txt").read().splitlines()  # 训练集图片的id
annos = json.loads(open(filedir).read())
classes_all = annos['types']  # 所有的交通标志类别
classes = open('../model_data/tt100k_classes.txt').read().splitlines()  # 要训练的类别
with open('../model_data/tt100k_classes_all.txt', 'w') as classes_all_file:  # 自动生成所有类别的列表
    for cls in classes_all:
        classes_all_file.write(cls + '\n')

file = open('../tt100k_train.txt', 'w')
for imgid in ids_train:
    img = annos['imgs'][imgid]
    imgpath = img['path']
    file.write(os.path.join(datadir, imgpath))
    objs = img['objects']
    for obj in objs:
        cls = obj['category']
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        bbox = obj['bbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
        b = (xmin, ymin, xmax, ymax)
        file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    file.write('\n')
file.close()
