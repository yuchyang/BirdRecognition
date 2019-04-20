import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from TEST import *

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 36
LR = 0.001          # 学习率

img_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/ImageRecognition',
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                            )

print(len(img_data))
data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE, shuffle=True)
print(len(data_loader))

from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module: Residual    Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    # 实现主module:ResNet34
    # ResNet34包含多个layer,每个layer又包含多个residual block
    # 用子module实现residual block , 用 _make_layer 函数实现layer
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
# resnet = models.resnet50()
# resnet = torch.load('test_mk3_0.7326666666666667.pkl')
# net = torchvision.models.densenet161(pretrained=False)
# net.classifier = nn.Linear(2208, 15)

net = models.resnet101(pretrained=True)
net.fc = nn.Linear(2048, 15)
print(net)
# for param in resnet.parameters():
#     param.requires_grad = False
# #但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
# for param in resnet.fc.parameters():
#     param.requires_grad = True
net.cuda()
# print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
standard = 0.85
Loss_list = []
train_accuracy_list = []
valid_accuracy_list = []
plt.ion()   # 画图
plt.show()
# training and testing
for epoch in range(EPOCH):
    correct = 0
    for step, (b_x, b_y) in enumerate(data_loader):   # 分配 batch data, normalize x when iterate train_loader
        x = b_x.cuda()
        y = b_y.cuda()
        output = net(x)# cnn output
        pred_y = torch.max(output, 1)[1].data.squeeze()
        loss = loss_func(output, y)     # cross entropy loss
        # print(loss.data)
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()           # apply gradients
        # print(pred_y)
        # print(y)
        if step%100 is 0:
            Loss_list.append(loss)
            x1 = range(0, len(Loss_list))
            y1 = Loss_list
            plt.subplot(2, 1, 1)
            plt.plot(x1, y1, 'o-')
            plt.xlabel('batch')
            plt.ylabel('Test loss')
            plt.pause(0.1)
    train_accuracy = test(net,'ImageRecognition')
    train_accuracy_list.append(train_accuracy)
    valid_accuracy = test(net,'test')
    valid_accuracy_list.append(valid_accuracy)
    print(train_accuracy)
    print(valid_accuracy)
    plt.subplot(2, 1, 2)
    x1 = range(0, len(train_accuracy_list))
    y1 = train_accuracy_list
    x2 = range(0, len(valid_accuracy_list))
    y2 = valid_accuracy_list
    plt.plot(x1, y1, 'g^', x2, y2, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.pause(0.1)
    if valid_accuracy > standard:
        standard = valid_accuracy
        torch.save(net, 'D://model//resnet101_{0}.pkl'.format(valid_accuracy))
plt.savefig('resnet101.png')
plt.show()