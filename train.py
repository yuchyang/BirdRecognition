import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import torch.optim as optim
from TEST import *
from utils import utils
from domain_adaptation import transfor_net


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 200           #
BATCH_SIZE = 42
LR = 0.001        # Learning rate

# img_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/ImageRecognition',
img_data = torchvision.datasets.ImageFolder('D:/IMAGE_TEST',

                                            transform=transforms.Compose([
                                                utils.Padding(),
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




# resnet = models.resnet50()
# net = torch.load('test_mk3_0.7326666666666667.pkl')
net = torchvision.models.resnet101(pretrained=False)
#

# net = models.resnet152(pretrained=False)
# net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
net.fc = nn.Linear(2048, 15)
# net.classifier = nn.Linear(2208, 15)
print(net)
# for param in net.parameters():
#     param.requires_grad = False
# for param in net.fc.parameters():
#     param.requires_grad = True
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=0.0005)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
standard = 0.80
Loss_list = []
train_accuracy_list = []
valid_accuracy_list = []
image_accuracy_list = []
plt.ion()   # 画图
plt.show()


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, iter_num):
        lr = self.init_lr * (1 + self.gamma * iter_num) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


# parameter_list = net.parameters()
# optimizer = optim.SGD(parameter_list, lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
# scheduler = INVScheduler(gamma=0.0003, decay_rate=0.75, init_lr=0.0003)
# group_ratios = [param_group["lr"] for param_group in optimizer.param_groups]


# training and testing
for epoch in range(EPOCH):
    correct = 0
    # valid_accuracy = test(net,'video recognition_test',show=False,shuffle=False)
    for step, (b_x, b_y) in enumerate(data_loader):   # 分配 batch data, normalize x when iterate train_loader

        # optimizer = scheduler.next_optimizer(group_ratios, optimizer, len(Loss_list))
        optimizer.zero_grad()

        x = b_x.cuda()
        y = b_y.cuda()
        output = net(x)# cnn output
        pred_y = torch.max(output, 1)[1].data.squeeze()
        loss = loss_func(output, y)     # cross entropy loss
        # print(loss.data)
        # optimizer.zero_grad()           # clear gradients for this training step
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
            plt.xlabel('')
            plt.ylabel('loss')
            plt.pause(0.1)
    # train_accuracy = test(net,'Mix_dataset',False,shuffle=False)
    # train_accuracy_list.append(train_accuracy)
    image_accuracy = test(net,'test',False,shuffle=False)
    image_accuracy_list.append(image_accuracy)

    # valid_accuracy = test(net,'video recognition_test',show=False,shuffle=False)
    # valid_accuracy_list.append(valid_accuracy)
    print(image_accuracy)
    # print(valid_accuracy)
    plt.subplot(2, 1, 2)
    x1 = range(0, len(image_accuracy_list))
    y1 = image_accuracy_list
    x2 = range(0, len(valid_accuracy_list))
    y2 = valid_accuracy_list
    plt.plot(x1, y1, 'g', x2, y2, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.pause(0.1)
    if image_accuracy > standard:
        standard = image_accuracy
        torch.save(net, 'D://model//resnet101_PADDING_{0}_{1}.pkl'.format(valid_accuracy,image_accuracy))
    if epoch is 100:
        plt.savefig('resnet101_IMAGE_only__BATCH_SIZE={0}.png'.format(BATCH_SIZE))
plt.savefig('resnet101_IMAGE_VIDEO_.png_epoch=200')
plt.show()