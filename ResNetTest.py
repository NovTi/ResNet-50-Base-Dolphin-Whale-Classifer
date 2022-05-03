import torch
from ResNet import ResNet50
from BottleNeck import BottleNeck


# def ResNet50(img_channels=3, num_classes=1000):
#     return ResNet(BottleNeck, [3, 4, 6, 3], img_channels, num_classes)


# print(torch.cuda.is_available())
# net = ResNet50(3, 2)
# x = torch.randn(4, 3, 224, 224)
# y = net(x)
# print(y.shape)
