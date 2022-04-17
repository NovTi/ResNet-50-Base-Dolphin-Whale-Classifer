import torch
from ResNet import ResNet
from BottleNeck import BottleNeck


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], img_channels, num_classes)

net = ResNet50()
x = torch.randn(4, 3, 224, 224)
y = net(x)
print(y.shape)
