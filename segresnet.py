import torch
import torch.nn as nn
import torch.nn.functional as F


class SegResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(SegResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Adjust the number of channels in the residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)  # Use a 1x1 convolution for the residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        out = self.dropout(out)
        return out


class SegResNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.0):
        super(SegResNet, self).__init__()
        # Encoder
        self.enc1 = SegResNetBlock(in_channels, 64, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = SegResNetBlock(64, 128, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = SegResNetBlock(128, 256, dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = SegResNetBlock(256, 512, dropout_rate)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = SegResNetBlock(512, 256, dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = SegResNetBlock(256, 128, dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = SegResNetBlock(128, 64, dropout_rate)

        # Output layer with two channels
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)

        # Decoder
        upconv3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat([enc3, upconv3], dim=1))
        upconv2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([enc2, upconv2], dim=1))
        upconv1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([enc1, upconv1], dim=1))

        # Output layer
        output = self.output_layer(dec1)

        return output


# Example usage:
in_channels = 3  # Assuming RGB input
num_classes = 2  # Two classes (kidney and background)
dropout_rate = 0.2  # Adjust as needed
seg_resnet = SegResNet(in_channels, num_classes, dropout_rate)
