import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def load_conv_bn(buf, start, conv_model, bn_model):
    """" load weights on conv & bn layers """
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start: start + num_b])); start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start: start + num_b])); start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start: start + num_b])); start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start: start + num_b])); start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start: start + num_w]).reshape_as(conv_model.weight)); start = start + num_w
    return start

 
def load_conv(buf, start, conv_model):
    """ load weights on conv layer """
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start: start + num_b])); start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start: start + num_w]).reshape_as(conv_model.weight)); start = start + num_w
    return start


class Conv2D(nn.Module):
    """ Conv2d, BatchNormalization, LeakyReLu + ZeroPadding2d """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(Conv2D, self).__init__()
        padding =(kernel_size - 1) // 2 if stride == 1 else 0
        self.stride = stride

        self.pad  = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x)))) if self.stride > 1 else self.relu(self.bn(self.conv(x)))


class Res_Block(nn.Module):
    """ Residual Block """
    def __init__(self, out_channel):
        super(Res_Block, self).__init__()
        self.conv = nn.Sequential( 
             Conv2D(2 * out_channel, out_channel, 1, 1),
             Conv2D(out_channel, 2 * out_channel, 3, 1))
 
    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        return x.add_(shortcut) # x + shortcut


class Darknet_53(nn.Module):
    def __init__(self):
        super(Darknet_53, self).__init__()

        self.cnn = nn.Sequential(OrderedDict([
              ('block0', Conv2D( 3, 32, 3, 1)),

              ('block1', nn.Sequential(
                         Conv2D(32, 64, 3, 2),
                         Res_Block(32))),

              ('block2', nn.Sequential(
                         Conv2D(64, 128, 3, 2),
                         *[Res_Block(64) for i in range(2)])),

              ('block3', nn.Sequential(
                         Conv2D(128, 256, 3, 2),
                         *[Res_Block(128) for i in range(8)])),

              ('block4', nn.Sequential(
                         Conv2D(256, 512, 3, 2),
                         *[Res_Block(256) for i in range(8)])),

              ('block5', nn.Sequential(
                         Conv2D(512, 1024, 3, 2),
                         *[Res_Block(512) for i in range(4)]))]))


    def forward(self, x):
        x       = self.cnn[0](x)
        x       = self.cnn[1](x)
        x       = self.cnn[2](x)
        route36 = self.cnn[3](x)
        route61 = self.cnn[4](route36)
        x       = self.cnn[5](route61)
        return route36, route61, x