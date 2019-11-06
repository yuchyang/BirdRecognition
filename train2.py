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
import model.backbone as backbone



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
        inputs = inputs.cuda()
        labels = labels.cuda()
        probabilities = model_instance(inputs)
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


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 200           #
BATCH_SIZE = 42
LR = 0.001        # Learning rate

img_data = torchvision.datasets.ImageFolder('D:/IMAGE2',
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



image_test_data = torchvision.datasets.ImageFolder('D:/IMAGE_TEST',
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


class DANNClassifier(nn.Module):
    def __init__(self, base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=31):
        super(DANNClassifier, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck_layer = nn.Linear(self.base_network.output_num(), bottleneck_dim)
            self.classifier_layer = nn.Linear(self.bottleneck_layer.out_features, class_num)
        else:
            self.classifier_layer = nn.Linear(self.base_network.output_num(), class_num)
        self.softmax = nn.Softmax()

        ## initialization
        if use_bottleneck:
            self.bottleneck_layer.weight.data.normal_(0, 0.005)
            self.bottleneck_layer.bias.data.fill_(0.1)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

        ## collect parameters
        if use_bottleneck:
            self.parameter_list = [{"params":self.base_network.parameters(), "lr":1},
                                {"params":self.bottleneck_layer.parameters(), "lr":10},
                            {"params":self.classifier_layer.parameters(), "lr":10}]

        else:
            self.parameter_list = [{"params":self.base_network.parameters(), "lr":1},
                            {"params":self.classifier_layer.parameters(), "lr":10}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        # softmax_outputs = self.softmax(outputs)
        return outputs

    def get_parameter_list(self):
        return self.parameter_list


    def set_train(self, mode):
        self.base_network.train(mode)
        self.is_train = mode

# resnet = models.resnet50()
# net = torch.load('test_mk3_0.7326666666666667.pkl')
net = DANNClassifier(base_net='ResNet101',use_bottleneck=True,bottleneck_dim=256,class_num=15)
parameter_list = net.get_parameter_list()
scheduler = INVScheduler(gamma=0.0003, decay_rate=0.75, init_lr=0.0003)

# net = models.resnet152(pretrained=False)
# net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
# net.fc = nn.Linear(2048, 15)
# net.classifier = nn.Linear(2208, 15)
print(net)
# for param in net.parameters():
#     param.requires_grad = False
# for param in net.fc.parameters():
#     param.requires_grad = True
net.cuda()
optimizer = optim.SGD(parameter_list, lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
standard = 0.80
Loss_list = []
train_accuracy_list = []
valid_accuracy_list = []
image_accuracy_list = []
plt.ion()   # 画图
plt.show()



loss_func = nn.CrossEntropyLoss()

# parameter_list = net.parameters()
# optimizer = optim.SGD(parameter_list, lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
# scheduler = INVScheduler(gamma=0.0003, decay_rate=0.75, init_lr=0.0003)
# group_ratios = [param_group["lr"] for param_group in optimizer.param_groups]
def train(model_instance, train_source_loader,test_source_loader ,num_iterations, batch_size, optimizer, lr_scheduler, group_ratios):
    train_accuracy_list = []
    valid_accuracy_list = []
    image_accuracy_list = []
    standard = 0.80
    plt.ion()  # 画图
    plt.show()
    num_batch_train_source = len(train_source_loader) - 1
    model_instance.set_train(True)
    ## train one iter
    print("start train...")
    for iter_num in range(num_iterations):
        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num)
        optimizer.zero_grad()
        if iter_num % num_batch_train_source == 0:
            iter_source = iter(train_source_loader)
        inputs_source, labels_source = iter_source.next()
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        output = model_instance(inputs_source)
        print(output.shape)
        print(labels_source.shape)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        # print(output)
        # print(labels_source)

        loss = loss_func(output,labels_source)
        loss.backward()
        if iter_num % 100 == 0:
            Loss_list.append(loss)
            x1 = range(0, len(Loss_list))
            y1 = Loss_list
            plt.subplot(2, 1, 1)
            plt.plot(x1, y1, 'o-')
            plt.xlabel('')
            plt.ylabel('loss')
            plt.pause(0.1)

        if iter_num % 500 == 0:
            # eval_result = evaluate(model_instance, test_target_loader)
            eval_result2 = evaluate(model_instance,test_source_loader)
            print('iteration number %s' % iter_num)
            # print(eval_result)
            print(eval_result2)
            image_accuracy_list.append(eval_result2['accuracy'])
            # valid_accuracy_list.append(eval_result['accuracy'])
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
            if eval_result2['accuracy'] > standard:
                standard = eval_result2['accuracy']
                model_instance.save_model(c_net_path='D://model//IMAGE_accuracy{0}_c_net'.format(standard),d_net_path='D://model//DANN_IMAGE_accuracy{0}_d_net'.format(standard))
    print("finish train.")


group_ratios = [param_group["lr"] for param_group in optimizer.param_groups]

train(net, data_loader, image_test_loader, 10000, batch_size=32, optimizer=optimizer,
      lr_scheduler=scheduler, group_ratios=group_ratios)

# training and testing
