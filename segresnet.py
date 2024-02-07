import torch
import torch.nn as nn

DROPOUT_PROB = 0.2

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=DROPOUT_PROB):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual
        out = self.relu(out)
        return out

class SegResNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, init_filters, num_blocks=[3, 4, 6, 3], dropout_prob=DROPOUT_PROB):
        super(SegResNet, self).__init__()
        self.in_channels = init_filters
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(init_filters, num_blocks[0], dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(init_filters * 2, num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(init_filters * 4, num_blocks[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(init_filters * 8, num_blocks[3], stride=2, dropout_prob=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(init_filters * 8, out_channels)

    def _make_layer(self, out_channels, num_blocks, stride=1, dropout_prob=0.0):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, dropout_prob))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define input tensor shape (batch_size, channels, spatial_dims)
input_tensor = torch.randn((32, 1, 256, 256)).to(device)

# Create SegResNet model
model = SegResNet(spatial_dims=2, in_channels=1, out_channels=2, init_filters=64, dropout_prob=0.2).to(device)

# Forward pass
output_tensor = model(input_tensor)

# Print model summary
print(model)

# Print output shape
print("Output shape:", output_tensor.shape)
