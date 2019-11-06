import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from utils import utils
from domain_adaptation import transfor_net


BATCH_SIZE = 10

def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def test(net,file,show,shuffle):
    print(net)
    net.eval()
    # net.set_train(False)
    test_data = torchvision.datasets.ImageFolder('D://IMAGE_TEST',
                                                 transform=transforms.Compose([
                                                     # utils.Padding(),
                                                     transforms.Resize(224),
                                                     # transforms.RandomCrop(224),
                                                     transforms.RandomHorizontalFlip(),
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
        print(x.data)
        # print(x.shape)
        # print(x.data.shape)
        output = net(x)  # cnn output
        # output = net.predict(x)
        # print(output.data)
        # print(output)
        print(output)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        # print(pred_y=)
        all += BATCH_SIZE
        # print(pred_y)
        # print(y)
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
    # net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
    # print(net.bottleneck_layer())
    # c_net = torch.load('D:/model/DANN_accuracy0.8743386243386243_c_net_0.9853333333333333.pkl')
    # model = transfor_net.DANN(base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=15,
    #                           hidden_dim=1024,
    #                           trade_off=1.0, use_gpu=True)
    # model.load_model('D:/model/DANN_accuracy0.8743386243386243_c_net_0.9853333333333333.pkl', 'D:/model/DANN_accuracy0.8743386243386243_d_net')
    # # print(net)
    # test(net,'TEST0',True,False)
    model = torchvision.models.densenet161(pretrained=False)
    model.classifier = torch.nn.Linear(2208, 15)
    model.load_state_dict(torch.load('densnet_0.94_dict.pth'))
    model.cuda()
    test(model, 'test', False, False)