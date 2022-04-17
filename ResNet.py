import torch
import torch.nn as nn



class ResNet(nn.Module):
    def __init__(self, bottleneck, layers, image_channels, class_nums):
        super().__init__()
        # initialize the in_channels after the first max pool layer
        self.in_channels = 64

        # conv1
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Res net bottleneck layers conv2, conv3, conv4, conv5
        # conv2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(bottleneck, layers[0], out_channels=64, stride=1)
        # conv3
        self.conv3 = self._make_layer(bottleneck, layers[1], out_channels=128, stride=2)
        # conv4
        self.conv4 = self._make_layer(bottleneck, layers[2], out_channels=256, stride=2)
        # conv5
        self.conv5 = self._make_layer(bottleneck, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, class_nums)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # what is this doing?
        x = self.fc(x)

        return x

    def _make_layer(self, bottleneck, block_nums, out_channels, stride):
        identity_downsample = None
        layers = []
        block_minus = 0
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )
            block_minus = 1
        layers.append(bottleneck(self.in_channels, out_channels, identity_downsample, stride=stride))
        self.in_channels = out_channels * 4

        for i in range(block_nums - block_minus):
            layers.append(bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # Why adding a *?????
