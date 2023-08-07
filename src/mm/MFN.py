import torch.nn as nn
import torch
import csv

path_hparams = "/home/houqijun/MobilefaceNet_torch/h_params.csv"

GDConv_size = (65, 4)

def readcsv(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([int(d) for d in row])

    return rows

class Conv_bn_prl(nn.Module):
    '''A model that consists of 2-d normal conv --> batchnorm --> prelu
    
    input of the model should be of shape [batch_size, n_channels, height, width]'''
    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = (1,1), padding = (0,0), groups = 1):
        super(Conv_bn_prl, self).__init__()
        # 2-d normal convolution
        self.conv2d = nn.Conv2d(in_channel, out_channels = out_channel,
            kernel_size = kernel_size, groups = groups, stride = stride, padding = padding, bias = False)
        # BatchNormalization
        self.bn = nn.BatchNorm2d(out_channel)
        # pReLU
        self.prelu = nn.PReLU(out_channel)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class BottleNeck(nn.Module):
    '''Module of BottleNeck Convolution: change demension-->seperable convolution-->batchnorm
     
    input of the model should be of shape [batch_size, n_channels, height, width]'''
    def __init__(self, in_channels, out_channels, res: bool = False, kernel=(3,3), stride = (2,2), padding = (1,1), t = 1):
        super(BottleNeck, self).__init__()
        # use 1*1 convolution to change the demension to t*in_channels
        c = t*in_channels
        self.conv11 = Conv_bn_prl(
            in_channel=in_channels, out_channel=c,
            kernel_size=(1,1), stride=(1,1), padding=(0,0)
        )
        # depthwise convolution(first step of a seperable convolution)
        self.conv_d = Conv_bn_prl(
            in_channel=c, out_channel=c,groups=c,
            kernel_size=kernel, stride=stride, padding=padding
        )
        # 1*1 convolution(second step of a seperable convolution)
        self.conv_combine = nn.Conv2d(
            in_channels=c, out_channels=out_channels, groups=1, 
            kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=False
        )
        # batch norm
        self.bn = nn.BatchNorm2d(out_channels)
        # residual connect or not
        self.residual = res

    def forward(self, x):
        if self.residual:
            out = self.conv11(x)
            out = self.conv_d(out)
            out = self.conv_combine(out)
            out = self.bn(out)
            out = out + x
            return out
        else:
            out = self.conv11(x)
            out = self.conv_d(out)
            out = self.conv_combine(out)
            out = self.bn(out)
            return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MFN(nn.Module):
    def __init__(self, in_channels) -> None:
        super(MFN, self).__init__()

        hparams = readcsv(path_hparams)
        final_layer_size = GDConv_size
        # normal conv layer
        self.conv33 = Conv_bn_prl(
            in_channel=in_channels, out_channel=hparams[0][1], kernel_size=(3,3),
            stride=(hparams[0][3],hparams[0][3]), padding=(1,1)
        )
        # depthwise conv 3*3 layer
        self.dconv33 = Conv_bn_prl(
            in_channel=hparams[0][1], out_channel=hparams[1][1], groups=hparams[0][1],
            kernel_size=(3,3), padding=(1,1), stride=(1,1)
        )
        # bottleneck t=2
        self.bottleneck1 = BottleNeck(
            in_channels=hparams[1][1], out_channels=hparams[2][1], t=hparams[2][0], 
            kernel=(3,3), stride=(hparams[2][3],hparams[2][3]), padding=(1,1)
        )
        # bottleneck t=2, with residual, repeat 4 times
        models_temp = []
        for _ in range(hparams[2][2]-1):
            models_temp.append(BottleNeck(
                in_channels=hparams[2][1], out_channels=hparams[2][1], t=hparams[2][0], res=True,
                kernel=(3,3), stride=(1,1), padding=(1,1)
            ))
        self.bottleneck_res_series1 = nn.Sequential(*models_temp)
        # bottleneck t=4
        self.bottleneck2 = BottleNeck(
            in_channels=hparams[2][1], out_channels=hparams[3][1], t=hparams[3][0], 
            kernel=(3,3), stride=(hparams[3][3],hparams[3][3]), padding=(1,1)
        )
        # bottleneck t=2, with residual, repeat 6 times
        models_temp = []
        for _ in range(hparams[4][2]):
            models_temp.append(BottleNeck(
                in_channels=hparams[3][1], out_channels=hparams[4][1], t=hparams[4][0], res=True,
                kernel=(3,3), stride=(1,1), padding=(1,1) 
            ))
        self.bottleneck_res_series2 = nn.Sequential(*models_temp)
        # bottleneck t=4
        self.bottleneck3 = BottleNeck(
            in_channels=hparams[4][1], out_channels=hparams[5][1], t=hparams[5][0], 
            kernel=(3,3), stride=(hparams[5][3],hparams[5][3]), padding=(1,1)
        )
        # bottleneck t=2, with residual, repeat 2 times
        models_temp = []
        for _ in range(hparams[6][2]):
            models_temp.append(BottleNeck(
                in_channels=hparams[5][1], out_channels=hparams[6][1], t=hparams[6][0], res=True,
                kernel=(3,3), stride=(1,1), padding=(1,1) 
            ))
        self.bottleneck_res_series3 = nn.Sequential(*models_temp)

        # 1*1 conv layer
        self.conv11 = Conv_bn_prl(
            in_channel=hparams[6][1], out_channel=hparams[7][1],
            kernel_size=(1,1), padding=(0,0), stride=(1,1)
        )
        # Global Depthwise Conv, which is a conv with kernel size == imagesize
        self.GDConv = nn.Conv2d(
            in_channels=hparams[7][1], out_channels=hparams[8][1], groups=hparams[8][1],
            kernel_size=final_layer_size, padding=(0,0), stride=(1,1), bias=False
        )
        # Flatten layer
        self.flatten = Flatten()
        self.out_dim = hparams[8][1]
        self.bn = nn.BatchNorm1d(self.out_dim)
    
    # @torch.no_grad()
    def forward(self, x):

        out = self.conv33(x)
        out = self.dconv33(out)
        out = self.bottleneck1(out)
        out = self.bottleneck_res_series1(out)
        out = self.bottleneck2(out)
        out = self.bottleneck_res_series2(out)
        out = self.bottleneck3(out)
        out = self.bottleneck_res_series3(out)
        out = self.conv11(out)
        out = self.GDConv(out)
        out = self.flatten(out)
        out = self.bn(out)

        return out
