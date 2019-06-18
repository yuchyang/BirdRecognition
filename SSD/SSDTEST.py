import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from utils import utils
from SSD.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite


BATCH_SIZE = 10


def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def test(net,file,show,shuffle):
    # net.eval()
    test_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/{0}'.format(file),
                                                 transform=transforms.Compose([
                                                     utils.Padding(),
                                                     transforms.Resize(300),
                                                     # transforms.RandomCrop(224),
                                                     # transforms.RandomHorizontalFlip(),
                                                     # transforms.CenterCrop(300),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((127, 127, 127), (128, 128, 128)),
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

        output = net(x)  # cnn output
        con = output[0]
        loc = output[1]
        batch = output[0].shape[0]
        box = output[0].shape[1]
        standard = 0.2
        for i in range(batch):
            all+=1
            for j in range(box):
                p = con[i][j]
                bird = p[3].float()
                if bird > standard:
                    standard = bird
                    print(bird)
                    correct+=1


    #     pred_y = torch.max(output, 1)[1].data.squeeze()
    #     print(pred_y.shape)
    #     all += BATCH_SIZE
    #     print(pred_y)
    #     print(y)
    #     for i in range(len(pred_y)):
    #         if pred_y[i] == y[i]:
    #             correct += 1
    #         else:
    #             if show is True:
    #                 utils.show_from_tensor(b_x[i])
    #     print('{0}/{1}'.format(correct, all))
    # print(correct/len(test_data))
    return correct/all

if __name__ == '__main__':
    # BATCH_SIZE = 1
    net = create_mobilenetv2_ssd_lite(21, is_test=True,onnx_compatible=True)
    net.load('D:/model/mb2-ssd-lite-mp-0_686.pth')
    net.cuda()
    accuracy=test(net,'video recognition_test',False,False)
    print(accuracy)