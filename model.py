import torch.nn as nn
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch.nn.init import xavier_normal_
from torchsummary import summary
# from mcdropout import MCDropout2D

IMG_HEIGHT = 512
IMG_WIDTH = 512


def makeLayerConfigure(start=64, num=4):
    encoder_ch = [1, 64, 128, 256, 512]
    decoder_ch = [512, 256, 128, 64]
    # encoder_ch.append(1)
    # doubles = start
    # encoder_ch.append(start)
    # for i in range(num - 1):
    #     doubles *= 2
    #     encoder_ch.append(doubles)

    # decoder_ch = encoder_ch[::-1][:-1]
    return encoder_ch, decoder_ch


class Block(nn.Module):
    def __init__(self, in_c, out_c, xavier, bc, retain_size=False):
        super().__init__()
        if retain_size:
            pad = 1
        else:
            pad = 0
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=1, padding=pad)
        if xavier:
            xavier_normal_(self.conv1.weight)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=pad)
        if xavier:
            xavier_normal_(self.conv2.weight)
        # batch normalization
        if bc:
            self.bc1 = nn.BatchNorm2d(out_c)
            self.bc2 = nn.BatchNorm2d(out_c)
        self.use_bc = bc

    def forward(self, x):
        if self.use_bc:
            x = self.relu(self.conv1(x))
            x = self.bc1(x)
            x = self.relu(self.conv2(x))
            x = self.bc2(x)
            return x
        else:
            return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, channels=[1, 64, 128, 256, 512], xavier=False, bc=False, dropout=None, retain_size=False):
        super().__init__()
        self.encBlocks = nn.ModuleList([
            Block(1, 64, xavier, bc, retain_size),
            Block(64, 128, xavier, bc, retain_size),
            Block(128, 256, xavier, bc, retain_size),
            Block(256, 512, xavier, bc, retain_size),
        ])
        
        # maxpooling layer
        self.pool = nn.MaxPool2d(2)
        self.dropout = dropout
        if dropout:
            # self.dropOutlayer = MCDropout2D(dropout)
            self.dropOutlayer = nn.Dropout2d(dropout)
    def forward(self, x):
        blockout = []
        for block in self.encBlocks[:-1]:
            x = block(x)
            blockout.append(x)
            x = self.pool(x)
            # if self.dropout:
            #     x = self.dropOutlayer(x)
        x = self.encBlocks[-1](x)
        if self.dropout:
            x = self.dropOutlayer(x)
        blockout.append(x)
        return blockout


class Decoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64], xavier=False, bc=False, retain_size=False):
        super().__init__()
        self.channels = channels
        
        # upconvolution layers
        layer_list = [
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ]
        
        # initialize weights with xavier
        for layer in layer_list:
            if xavier:
                layer.weight.data = xavier_normal_(layer.weight.data)

        # making upconv as modulelist
        self.upconv = nn.ModuleList(layer_list)
        
        # making decBlocks as modulelist
        self.decBlocks = nn.ModuleList([
            Block(512, 256, xavier, bc, retain_size),
            Block(256, 128, xavier, bc, retain_size),
            Block(128, 64, xavier, bc, retain_size),
        ])
        

    def forward(self, x, encf):
        for i in range(len(self.channels) - 1):
            # upconvolution
            x = self.upconv[i](x)
            # crop the encoder feature map and concatenate
            encfeature = self.crop(encf[i], x)
            x = torch.cat([x, encfeature], dim=1)
            # pass through decoder block
            x = self.decBlocks[i](x)
        return x

    # crop the encoder feature map
    def crop(self, encfeature, x):
        (batch_size, ch, H, W) = x.shape
        encfeature = CenterCrop([H, W])(encfeature)
        return encfeature


class UNet(nn.Module):
    def __init__(self, start_ch, num_ch, use_xavier=True, use_batchNorm=True, dropout=0.4, retain_size=True,
                nbCls=1, retainDim=True, outsize=(IMG_HEIGHT, IMG_HEIGHT)):
        super().__init__()
        (encCh, decCh) = makeLayerConfigure(start_ch, num_ch)
        self.encoder = Encoder(encCh, xavier=use_xavier, bc=use_batchNorm, dropout=dropout, retain_size=retain_size)
        self.decoder = Decoder(decCh, xavier=use_xavier, bc=use_batchNorm, retain_size=retain_size)
        self.head = nn.Conv2d(decCh[-1], nbCls, 1)
        self.retainDim = retainDim
        self.outsize = outsize
        self.retainSize = retain_size

    def forward(self, x):
        # encoder feature map
        encf = self.encoder(x)
        # decoder feature map
        decf = self.decoder(encf[-1], encf[::-1][1:])
        # output layer
        out = self.head(decf)
        
        # retain size
        if self.retainSize:
            return out
        # retain dimension
        if self.retainDim:
            out = F.interpolate(out, self.outsize)
        return out


if __name__ == '__main__':
    unet = UNet(64, 5, use_xavier=True, use_batchNorm=True, dropout=0.4, retainDim=True, retain_size=True)
    tmp = torch.rand((1, 1, IMG_HEIGHT, IMG_WIDTH))
    print(tmp.shape)
    out = unet(tmp)
    print(out.shape)
    print(summary(unet, input_size=(1, IMG_HEIGHT, IMG_WIDTH), device='cpu'))