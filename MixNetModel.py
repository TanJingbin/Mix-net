import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np


# class mywish(nn.Module):
#     def __init__(self, beta, positive_flag=True):
#         super(mywish, self).__init__()
#         self.positive_flag = positive_flag
#         self.beta = beta
#
#     def forward(self, x):
#         x = x.clone().detach()
#         if self.positive_flag:
#             out = torch.where(x <= -5, self.beta * x, torch.clamp(x, min=0, max=6))
#         else:
#             out = x / (1 + torch.exp(-x))
#         out = torch.tensor(out)
#         return out
#
#
# # 设置自定义输出值条件定义：
# class myswish_function(nn.Module):
#     def __init__(self, beta):
#         super(myswish_function, self).__init__()
#         self.beta = beta
#
#     def forward(self, x):
#         x = x.clone().detach()
#         f0 = mywish(beta=self.beta, positive_flag=True)
#         out0 = f0(x)
#         f1 = mywish(beta=self.beta, positive_flag=False)
#         out1 = f1(x)
#         out = torch.where((x < 6) & (x > -5), out1, out0)
#         # out = torch.tensor(out)
#         return out


# WS
def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)


# SE
class SeModule(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        z = x * y.expand_as(x)
        return z


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = SeModule(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse + U_sse


# shuffle_channel
def channel_shuffle(x, groups=3):
    batchsize, num_channels, height, width = x.data.size()
    if num_channels % groups:
        return x
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


############################################################################
# This class is responsible for a single depth-wise separable convolution step
class dilated_downsample_up(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_downsample_up, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(4, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_downsample_down(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_downsample_down, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(4, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_downsample_pool(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_downsample_pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel, stride, padding, groups=1, dilation=dilation),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),

        )

    def forward(self, x):
        x = self.layer(x)
        return x


# This class is responsible for a single depth-wise separable De-convolution step
class dilated_upsample_up(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_upsample_up, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(4, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_upsample_down(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_upsample_down, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(4, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_upsample_pool(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_upsample_pool, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, kernel, stride, padding, groups=1, dilation=dilation),
        )
        self.bn = nn.BatchNorm2d(C_out)
        self.lRelu = nn.LeakyReLU()

    def forward(self, x, last=False):
        x = self.layer(x)
        if last == False:
            x = self.lRelu(self.bn(x))
        return x


# This class incorporates the convolution steps in parallel with different dilation rates and
# concatenates their output. This class is called at each level of the U-NET encoder
class BasicBlock_downsample_pool(nn.Module):
    def __init__(self, c1, c2, k1, k2, s1, s2, p1, p2):
        super(BasicBlock_downsample_pool, self).__init__()

        self.d1 = dilated_downsample_pool(c1, c2, k1, s1, p1, dilation=1)
        self.d2 = dilated_downsample_pool(c2, c2, k2, s2, p2, dilation=1)
        self.d3 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # x1 = self.d1(x)
        # x2 = self.d2(x1)
        x3 = self.d3(x)

        return x3


class BasicBlock_upsample_pool(nn.Module):
    def __init__(self, c1, c2, k1, k2, s1, s2, p1, p2):
        super(BasicBlock_upsample_pool, self).__init__()

        self.d1 = dilated_upsample_pool(c1, c2, k1, s1, p1, dilation=1)
        self.d2 = dilated_upsample_pool(c2, c2, k2, s2, p2, dilation=1)
        # self.d3 = nn.MaxUnpool2d(2, stride=2)
        self.d3 = nn.ConvTranspose2d(c1, c2, 2, 2, 0)

    def forward(self, x, y=None):
        # x1 = self.d1(x)
        # x2 = self.d2(x1)
        x3 = self.d3(x)
        # x3 = nn.functional.interpolate(x2, (32, 64), mode='bilinear', align_corners=True)
        x_result = x3

        if y is not None:
            return torch.cat([x_result, y], dim=1)
        return x_result


class BasicBlock_downsample_up(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_downsample_up, self).__init__()
        self.d1 = dilated_downsample_up(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_downsample_up(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_downsample_up(c1, c2, k3, s3, p3, dilation=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        return torch.cat([x1, x2, x3], dim=1)


class BasicBlock_downsample_down(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_downsample_down, self).__init__()
        self.d1 = dilated_downsample_down(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_downsample_down(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_downsample_down(c1, c2, k3, s3, p3, dilation=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        return torch.cat([x1, x2, x3], dim=1)


# This class incorporates the De-convolution steps in parallel with different dilation rates and
# concatenates their output. This class is called at each level of the U-NET Decoder

class BasicBlock_upsample_down(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_upsample_down, self).__init__()

        self.d1 = dilated_upsample_down(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_upsample_down(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_upsample_down(c1, c2, k3, s3, p3, dilation=1)

        self.resize = dilated_upsample_down(c2, c2, 2, 1, 0, 1)

    def forward(self, x, y=None):
        x1 = self.d1(x)
        x2 = self.resize(self.d2(x))
        x3 = self.d3(x)

        x_result = torch.cat([x1, x2, x3], dim=1)

        if y is not None:
            return torch.cat([x_result, y], dim=1)
        return x_result


class BasicBlock_upsample_up(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_upsample_up, self).__init__()

        self.d1 = dilated_upsample_up(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_upsample_up(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_upsample_up(c1, c2, k3, s3, p3, dilation=1)

        self.resize = dilated_upsample_up(c2, c2, 2, 1, 0, 1)

    def forward(self, x, y=None):
        x1 = self.d1(x)
        x2 = self.resize(self.d2(x))
        x3 = self.d3(x)

        x_result = torch.cat([x1, x2, x3], dim=1)

        if y is not None:
            return torch.cat([x_result, y], dim=1)
        return x_result


class Dilated_UNET(nn.Module):
    def __init__(self):
        super(Dilated_UNET, self).__init__()
        # Encoder1 - with shape output commented for each step
        self.d2 = BasicBlock_downsample_up(3, 12, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3, p3=1)  # 36,128,128
        self.d3 = BasicBlock_downsample_up(36, 36, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3, p3=1)  # 108,64,64
        self.d4 = BasicBlock_downsample_down(108, 108, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3,
                                             p3=1)
        self.d5 = BasicBlock_downsample_down(324, 324, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3,
                                             p3=1)

        # Decoder1 - with shape output commented for each step
        self.u1 = BasicBlock_upsample_down(972, 108, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u2 = BasicBlock_upsample_down(648, 36, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u3 = BasicBlock_upsample_up(216, 12, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u4 = BasicBlock_upsample_up(72, 4, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)

        # Encoder2 - with shape output commented for each step
        self.pd1 = BasicBlock_downsample_pool(3, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pd2 = BasicBlock_downsample_pool(3, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pd3 = BasicBlock_downsample_pool(3, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pd4 = BasicBlock_downsample_pool(3, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)

        # Decoder2 - with shape output commented for each step
        self.pu1 = BasicBlock_upsample_pool(3, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pu2 = BasicBlock_upsample_pool(6, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pu3 = BasicBlock_upsample_pool(6, 3, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)
        self.pu4 = BasicBlock_upsample_pool(6, 4, k1=3, k2=3, s1=1, s2=1, p1=1, p2=1)

        # Classifier
        self.classifier = nn.Conv2d(16, 20, 3, 1, 1)

        # Dropout layers
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.15)

    def forward(self, x):
        x1 = x

        down1 = self.drop1(self.d2(x))
        down2 = self.drop1(self.d3(down1))
        down3 = self.drop1(self.d4(down2))
        down4 = self.drop1(self.d5(down3))

        up1 = self.drop2(self.u1(down4, down3))
        up2 = self.drop2(self.u2(up1, down2))
        up3 = self.drop2(self.u3(up2, down1))
        up4 = self.drop2(self.u4(up3))

        pdown1 = self.pd1(x1)
        pdown2 = self.pd2(pdown1)
        pdown3 = self.pd3(pdown2)
        pdown4 = self.pd4(pdown3)

        pup1 = self.pu1(pdown4, pdown3)
        pup2 = self.pu2(pup1, pdown2)
        pup3 = self.pu3(pup2, pdown1)
        pup4 = self.pu4(pup3)

        cf = torch.cat([up4, pup4], dim=1)
        return self.classifier(cf)
