import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from utils import utils

BATCH_SIZE = 10


def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def test(net,file,show,shuffle):
    net.eval()
    test_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/{0}'.format(file),
                                                 transform=transforms.Compose([
                                                     transforms.Resize(224),
                                                     # transforms.RandomCrop(224),
                                                     # transforms.RandomHorizontalFlip(),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
                                                )
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=shuffle)

    a = []
    print(test_data)

    correct = 0
    all = 0
    for step, (b_x, b_y) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
        x = b_x.cuda()
        y = b_y.cuda()
        print(x.data.shape)
        output = net(x)  # cnn output
        print(output.data)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        print(pred_y.shape)
        all += BATCH_SIZE
        print(pred_y)
        print(y)
        for i in range(len(pred_y)):
            if pred_y[i] == y[i]:
                correct += 1
            else:
                if show is True:
                    utils.show_from_tensor(b_x[i])
        print('{0}/{1}'.format(correct, all))
    print(correct/len(test_data))
    return correct/len(test_data)

if __name__ == '__main__':
    # BATCH_SIZE = 1
    net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
    # net = torch.load('densnet_0.904.pkl')
    test(net,'video recognition',False,False)