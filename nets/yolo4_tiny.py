import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet53_tiny import darknet53_tiny


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
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53_tiny(None)

        self.conv_for_P4 = BasicConv(256, 128, 1)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 128)

        self.upsample = Upsample(128, 64)
        self.yolo_headP3 = yolo_head([128, num_anchors * (5 + num_classes)], 192)

    def forward(self, x):
        #  backbone
        feat1, feat2 = self.backbone(x)  # 52,26 128 256
        P4 = self.conv_for_P4(feat2)  # 256-128
        out0 = self.yolo_headP4(P4)

        P4_Upsample = self.upsample(P4)  # 128-64
        P3 = torch.cat([feat1, P4_Upsample], axis=1)  # 192
        out1=self.yolo_headP3(P3)

        return out0, out1
