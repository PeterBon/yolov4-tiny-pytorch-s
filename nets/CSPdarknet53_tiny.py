import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
from nets.attention_layers import SEModule, CBAM

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# -------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+LeakyReLU
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.attention1 = CBAM(out_channels // 2)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.attention2 = CBAM(out_channels)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        route = x

        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = self.attention1(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        feat = x

        x = self.attention2(x)
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        return x, feat


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.resblock_body1 = Resblock_body(32, 32)

        self.resblock_body2 = Resblock_body(64, 64)
        self.resblock_body3 = Resblock_body(128, 128)

        self.conv2 = BasicConv(256, 256, kernel_size=3)

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)  # 416->208
        x = self.conv2(x)  # 208->104
        x, _ = self.resblock_body1(x)  # 104->52,104
        x, feat1 = self.resblock_body2(x)  # 52->26,52
        # x, feat1 = self.resblock_body3(x)  # 26->13,26
        x = self.conv3(x)  # 26->26
        feat2 = x
        return feat1, feat2  # 52,26



def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
