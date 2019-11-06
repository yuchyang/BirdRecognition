import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import copy
from torchvision import models


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        ## set base network
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.high_dim = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        low_features = x
        x = self.avgpool(x)
        high_features = x.view(x.size(0), -1)
        return low_features, high_features

    def output_dim(self):
        return self.high_dim


class Bottleneck(nn.Module):
    def __init__(self, feature_dim=2048, bottleneck_dim=256):
        super(Bottleneck, self).__init__()
        ## initialization
        self.bottleneck_layer = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck_layer.weight.data.normal_(0, 0.005)
        self.bottleneck_layer.bias.data.fill_(0.1)

    def forward(self, inputs):
        features = self.bottleneck_layer(inputs)
        return features


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=31):
        super(FeatureClassifier, self).__init__()
        self.classifier_layer = nn.Linear(feature_dim, class_num)
        self.softmax = nn.Softmax()

        ## initialization
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, inputs):
        outputs = self.classifier_layer(inputs)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs


class HighDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(HighDiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        #self.grl_layer = grl.GradientReverseLayer()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)

        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)

        self.forward_count = 0

    def forward(self, inputs):
        self.forward_count += 1
        grl_layer = grl.InstantGRLayer(iter_num=self.forward_count)
        outputs = grl_layer(inputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs


class LowDiscriminator(nn.Module):
    def __init__(self, input_channel=1, hidden_channel=64):
        super(LowDiscriminator, self).__init__()

        self.forward_count = 0
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.feature = nn.Sequential(
                nn.Conv2d(input_channel, self.hidden_channel, 3, 1, 1),            
                nn.BatchNorm2d(self.hidden_channel),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.ReLU(0.2),
                nn.MaxPool2d(2,2),

                nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, 1, 1),         
                nn.BatchNorm2d(self.hidden_channel),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2,2),

                #nn.Conv2d(self.hidden_channel, self.hidden_channel//2, 2, 1, 1),           
                #nn.BatchNorm2d(self.hidden_channel//2),
                #nn.LeakyReLU(0.2, inplace=True),
                #nn.MaxPool2d(2,2),
                )

        self.classifier_s = nn.Sequential(
                nn.Linear(self.hidden_channel, 1), 
                nn.Sigmoid()
                ) 

    def forward(self, inputs):
        self.forward_count += 1
        grl_layer = grl.InstantGRLayer(iter_num=self.forward_count)
        outputs = grl_layer(inputs)
        features = self.feature(outputs)
        features = features.view(-1,self.hidden_channel)
        probabilities = self.classifier_s(features)
        return probabilities


class DANN(object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31, hidden_dim=1024, input_channel=2048, hidden_channel=512, high_trade_off=1.0, low_trade_off=0.5, dis_trade_off=0.01, use_gpu=True, writer=None):

        self.f_net = FeatureExtractor()
        high_feature_dim = self.f_net.output_dim()
        self.bottleneck = Bottleneck(feature_dim=high_feature_dim, bottleneck_dim=bottleneck_dim)
        feature_dim = bottleneck_dim
        self.c_net = FeatureClassifier(feature_dim=feature_dim, class_num=class_num)
        self.d_net_high = HighDiscriminator(feature_dim, hidden_dim)
        self.d_net_low = LowDiscriminator(input_channel=input_channel, hidden_channel=hidden_channel)
        self.high_trade_off = high_trade_off
        self.low_trade_off = low_trade_off
        self.dis_trade_off = dis_trade_off
        self.use_gpu = use_gpu
        self.is_train = False
        self.writer = writer

        self.sigmoid = nn.Sigmoid()
        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.bottleneck = self.bottleneck.cuda()
            self.c_net = self.c_net.cuda()
            self.d_net_high = self.d_net_high.cuda()
            self.d_net_low = self.d_net_low.cuda()
    
    def get_loss(self, inputs, labels_source, epoch):
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        distance_criterion = nn.L1Loss()
        batch_size = inputs.size(0) // 2

        low_features, high_features  = self.f_net(inputs)
        high_features = self.bottleneck(high_features)
        outputs, probabilities = self.c_net(high_features)
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)//2), labels_source)
        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            }, epoch)

        high_dc_outputs = self.d_net_high(high_features)
        low_dc_outputs = self.d_net_low(low_features)
        dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
        if self.use_gpu:
            dc_target = dc_target.cuda()
        high_transfer_loss = transfer_criterion(high_dc_outputs, dc_target) * self.high_trade_off
        low_transfer_loss = transfer_criterion(low_dc_outputs, dc_target) * self.low_trade_off
        high_dc_outputs_src = high_dc_outputs.narrow(0, 0, inputs.size(0)//2)
        high_dc_outputs_tgt = high_dc_outputs.narrow(0, inputs.size(0)//2, inputs.size(0)//2)
        low_dc_outputs_src = low_dc_outputs.narrow(0, 0, inputs.size(0)//2)
        low_dc_outputs_tgt = low_dc_outputs.narrow(0, inputs.size(0)//2, inputs.size(0)//2)
        self.writer.add_scalars('probabilities/domainclassifier', {
            'high-src': high_dc_outputs_src.mean(),
            'high-tgt': high_dc_outputs_tgt.mean(),
            'low-src': low_dc_outputs_src.mean(),
            'low-tgt': low_dc_outputs_tgt.mean(),
            }, epoch)
        self.writer.add_scalars('data/loss', {
            'domain_high_loss': high_transfer_loss,
            'domain_low_loss': low_transfer_loss,
            }, epoch)
        transfer_loss = high_transfer_loss + low_transfer_loss
        total_loss = transfer_loss + classifier_loss
        return total_loss


    def predict(self, inputs):
        _, high_features = self.f_net(inputs)
        features = self.bottleneck(high_features)
        _, softmax_outputs = self.c_net(features)
        return softmax_outputs


    def get_parameter_list(self):
        parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.bottleneck.parameters(), 'lr': 10},
                {'params': self.c_net.parameters(), 'lr': 10},
                {'params': self.d_net_high.parameters(), 'lr': 10},
                {'params': self.d_net_low.parameters(), 'lr': 10},
                ]
        return parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.bottleneck.train(mode)
        self.c_net.train(mode)
        self.d_net_high.train(mode)
        self.d_net_low.train(mode)
        self.is_train = mode
 
