import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from TEST import *
from domain_adaptation import transfor_net
import torch.optim as optim

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 36
LR = 0.0001
batch_ratio = 0.5   # video_image/all

img_train_address = 'D:/IMAGE2'
image_test_address = 'C:/Users/lyyc/Desktop/BirdRecognition/image_test'
video_train_address = 'C:/Users/lyyc/Desktop/BirdRecognition/video recognition_train'
video_test_address ='C:/Users/lyyc/Desktop/BirdRecognition/video recognition_test'
base_net_address = 'D:/model/resnet101_0.9606666666666667.pkl'

# img_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/image_train',
img_data = torchvision.datasets.ImageFolder(img_train_address,
                                               transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                            )

print(len(img_data))
data_loader = torch.utils.data.DataLoader(img_data, batch_size=16, shuffle=True)
print(len(data_loader))

video_data = torchvision.datasets.ImageFolder(video_train_address,
# video_data = torchvision.datasets.ImageFolder('D:/IMAGE2',
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                            )

data_loader2 = torch.utils.data.DataLoader(video_data, batch_size=16, shuffle=True)

print(len(video_data))
# data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE*(1-batch_ratio), shuffle=True)
print(len(data_loader2))

test_data = torchvision.datasets.ImageFolder(video_test_address,
                                            transform=transforms.Compose([
                                                transforms.Resize(224),
                                                # transforms.RandomCrop(224),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                            )

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

print(len(test_data))
# data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE*(1-batch_ratio), shuffle=True)
print(len(test_loader))

image_test_data = torchvision.datasets.ImageFolder(image_test_address,
                                            transform=transforms.Compose([
                                                transforms.Resize(224),
                                                # transforms.RandomCrop(224),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                            )

image_test_loader = torch.utils.data.DataLoader(image_test_data, batch_size=32, shuffle=False)

print(len(image_test_data))
# data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE*(1-batch_ratio), shuffle=True)
print(len(image_test_loader))


from torch import nn
import torch as t
from torch.nn import functional as F



# resnet = models.resnet50()
# resnet = torch.load('test_mk3_0.7326666666666667.pkl')
# net = torchvision.models.densenet161(pretrained=False)
# net.classifier = nn.Linear(2208, 15)

net = models.resnet101(pretrained=False)
# net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
# net.fc = nn.Linear(2048, 15)
# print(net)
# for param in net.parameters():
#     param.requires_grad = False
# for param in net.fc.parameters():
#     param.requires_grad = True
net.cuda()
# print(net)
# optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
standard = 0.6
Loss_list = []
train_accuracy_list = []
valid_accuracy_list = []
plt.ion()   # 画图
plt.show()
# training and testing

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)
        self.Linear_layer = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.resnet_layer(x)

        x = self.transion_layer(x)

        x = self.pool_layer(x)

        x = x.view(x.size(0), -1)

        x = self.Linear_layer(x)



# for epoch in range(EPOCH):
#     correct = 0
#     # valid_accuracy = test(net,'video recognition_test',show=False,shuffle=False)
#     for step, (b_x, b_y) in enumerate(data_loader):   # 分配 batch data, normalize x when iterate train_loader
#         x = b_x.cuda()
#         y = b_y.cuda()
#         output = net(x)# cnn output
#         pred_y = torch.max(output, 1)[1].data.squeeze()
#         loss = loss_func(output, y)     # cross entropy loss
#         # print(loss.data)
#         optimizer.zero_grad()           # clear gradients for this training step
#         loss.backward()                 # backpropagation, compute gradients
#         optimizer.step()           # apply gradients
#         # print(pred_y)
#         # print(y)
#         if step%100 is 0:
#             Loss_list.append(loss)
#             x1 = range(0, len(Loss_list))
#             y1 = Loss_list
#             plt.subplot(2, 1, 1)
#             plt.plot(x1, y1, 'o-')
#             plt.xlabel('')
#             plt.ylabel('loss')
#             plt.pause(0.1)
#     train_accuracy = test(net,'ImageRecognition',False,shuffle=False)
#     train_accuracy_list.append(train_accuracy)
#     valid_accuracy = test(net,'video recognition_test',show=False,shuffle=False)
#     valid_accuracy_list.append(valid_accuracy)
#     print(train_accuracy)
#     print(valid_accuracy)
#     plt.subplot(2, 1, 2)
#     x1 = range(0, len(train_accuracy_list))
#     y1 = train_accuracy_list
#     x2 = range(0, len(valid_accuracy_list))
#     y2 = valid_accuracy_list
#     plt.plot(x1, y1, 'g^', x2, y2, 'b')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.pause(0.1)
#     if valid_accuracy > standard:
#         standard = valid_accuracy
#         torch.save(net, 'D://model//resnet101_MIX_DATSET_LR = 0.0001_{0}.pkl'.format(valid_accuracy))

def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.long()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    model_instance.set_train(ori_train_state)
    _, predict = torch.max(all_probs, 1)

    accuracy = float(torch.sum(torch.squeeze(predict).long() == all_labels)) / float(all_labels.size()[0])
    return {'accuracy':accuracy}


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


def train(model_instance, train_source_loader, train_target_loader, test_target_loader,test_source_loader ,num_iterations, batch_size, optimizer, lr_scheduler, group_ratios):
    train_accuracy_list = []
    valid_accuracy_list = []
    image_accuracy_list = []
    standard = 0.80
    plt.ion()  # 画图
    plt.show()
    num_batch_train_source = len(train_source_loader) - 1
    num_batch_train_target = len(train_target_loader) - 1
    model_instance.set_train(True)
    ## train one iter
    print("start train...")
    for iter_num in range(num_iterations):
        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num)
        optimizer.zero_grad()
        if iter_num % num_batch_train_source == 0:
            iter_source = iter(train_source_loader)
        if iter_num % num_batch_train_target == 0:
            iter_target = iter(train_target_loader)
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if model_instance.use_gpu:
            inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(),\
                                                          labels_source.cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(
                labels_source)
        loss = train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, iter_num)
        if iter_num % 100 == 0:
            Loss_list.append(loss)
            x1 = range(0, len(Loss_list))
            y1 = Loss_list
            plt.subplot(2, 1, 1)
            plt.plot(x1, y1, 'o-')
            plt.xlabel('')
            plt.ylabel('loss')

        if iter_num % 500 == 0 and iter_num is not 0:
            eval_result = evaluate(model_instance, test_target_loader)
            eval_result2 = evaluate(model_instance,test_source_loader)
            print('iteration number %s' % iter_num)
            print(eval_result)
            print(eval_result2)
            image_accuracy_list.append(eval_result2['accuracy'])
            valid_accuracy_list.append(eval_result['accuracy'])
            plt.subplot(2, 1, 2)
            x1 = range(0, len(image_accuracy_list))
            y1 = image_accuracy_list
            x2 = range(0, len(valid_accuracy_list))
            y2 = valid_accuracy_list
            plt.plot(x1, y1, 'g', x2, y2, 'b')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.pause(0.1)
            # writer.add_scalars('data/accu', {
            #     'tgt accu': eval_result['accuracy'],
            # }, iter_num)
            if len(valid_accuracy_list)%100 is 0 and len(valid_accuracy_list) is not 0:
                plt.savefig('DANN_{0}.png'.format(len(valid_accuracy_list)))

            if eval_result2['accuracy'] > standard:
                standard = eval_result2['accuracy']
                model_instance.save_model(c_net_path='DANN_IMAGE_accuracy{0}_c_net'.format(standard),d_net_path='DANN_IMAGE_accuracy{0}_d_net'.format(standard))
    print("finish train.")


def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, epoch):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source, epoch)
    total_loss.backward()
    optimizer.step()
    return total_loss

# def train_batch(model_instance, inputs_source, labels_source, inputs_target, labels_target, optimizer, epoch):
#     inputs = torch.cat((inputs_source, inputs_target), dim=0)
#     labels = torch.cat((labels_source, labels_target), dim=0)
#     total_loss = model_instance.get_loss(inputs, labels, epoch)
#     print(total_loss)
#     total_loss.backward()
#     optimizer.step()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':

    base_net = torch.load(base_net_address)
    base_net.fc = Identity()
    # model = transfor_net.DANN(base_net=base_net,use_bottleneck=False,trade_off=1)
    model = transfor_net.DANN(base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=15, hidden_dim=1024,
                          trade_off=1.0, use_gpu=True)
    # Set optimizer
    parameter_list = model.get_parameter_list()
    optimizer = optim.SGD(parameter_list, lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = INVScheduler(gamma=0.0002, decay_rate=0.75, init_lr=0.0003)
    group_ratios = [param_group["lr"] for param_group in optimizer.param_groups]
    # Train model
    train(model,data_loader,data_loader2,test_loader,image_test_loader,1000000, batch_size=32, optimizer=optimizer, lr_scheduler=scheduler, group_ratios=group_ratios)
    # train(model,data_loader,data_loader2,test_loader,1000000,batch_size=32,optimizer=optimizer)
    # plt.savefig('resnet101_MIX_DATSET_LR = 0.0001.png')
    # plt.show()