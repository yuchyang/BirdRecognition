import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
BATCH_SIZE = 10
def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def test(net,file):
    test_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/{0}'.format(file),
                                                 transform=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.RandomCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     # transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
                                                )
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    a = []
    print(len(data_loader))

    correct = 0
    all = 0
    for step, (b_x, b_y) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
        x = b_x.cuda()
        y = b_y.cuda()
        output = net(x)  # cnn output
        pred_y = torch.max(output, 1)[1].data.squeeze()
        all += BATCH_SIZE
        for i in range(len(pred_y)):
            print(i)
            if pred_y[i] == y[i]:
                correct += 1
        print('{0}/{1}'.format(correct, all))
    print(correct/len(test_data))
    return correct/len(test_data)

if __name__ == '__main__':
    # BATCH_SIZE = 1
    net = torch.load('densnet_0.934.pkl')
    test(net,'TEST3')