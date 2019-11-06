import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import model.backbone as backbone
import copy


class FeatureExtractor(nn.Module):
    def __init__(self, base_net='ResNet50'):
        super(FeatureExtractor, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()

    def forward(self, inputs):
        features = self.base_network(inputs)
        return features

    def output_dim(self):
        return self.base_network.output_num()


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


class Discriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Discriminator, self).__init__()

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


class DANN(object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31, hidden_dim=1024, cla1_trade_off=1.0, cla2_trade_off=0.1, dis_trade_off=0.01, use_gpu=True, more_since=10000, writer=None):

        self.f_net = FeatureExtractor(base_net=base_net)
        feature_dim = self.f_net.output_dim()
        self.bottleneck = Bottleneck(feature_dim=feature_dim, bottleneck_dim=bottleneck_dim)
        feature_dim = bottleneck_dim
        self.c_net = FeatureClassifier(feature_dim=feature_dim, class_num=class_num)
        self.d_net_1 = Discriminator(feature_dim, hidden_dim)
        self.d_net_2 = Discriminator(feature_dim, hidden_dim)
        self.cla1_trade_off = cla1_trade_off
        self.cla2_trade_off = cla2_trade_off
        self.dis_trade_off = dis_trade_off
        self.use_gpu = use_gpu
        self.is_train = False
        self.start_second = False
        self.more_since = more_since
        self.writer = writer

        self.sigmoid = nn.Sigmoid()
        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.bottleneck = self.bottleneck.cuda()
            self.c_net = self.c_net.cuda()
            self.d_net_1 = self.d_net_1.cuda()
            self.d_net_2 = self.d_net_2.cuda()
    
    def get_loss(self, inputs, labels_source, epoch):
        if not self.start_second and epoch > self.more_since:
            self.d_net_2 = copy.deepcopy(self.d_net_1)
            self.start_second = True
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        distance_criterion = nn.L1Loss()
        batch_size = inputs.size(0) // 2

        features = self.f_net(inputs)
        features = self.bottleneck(features)
        outputs, probabilities = self.c_net(features)
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)//2), labels_source)
        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            }, epoch)

        dc_outputs = self.d_net_1(features)
        dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
        if self.use_gpu:
            dc_target = dc_target.cuda()
        transfer_loss = transfer_criterion(dc_outputs, dc_target) * self.cla1_trade_off
        self.writer.add_scalars('data/loss', {
            'domain_cla1_loss': transfer_loss,
            }, epoch)
        if self.start_second:
            dc_outputs_2 = self.d_net_2(features)
            transfer_loss_2 = transfer_criterion(dc_outputs_2, dc_target) * self.cla2_trade_off
            transfer_loss += transfer_loss_2
            distance_loss = 0
            for paramA, paramB in zip(self.d_net_1.parameters(), self.d_net_2.parameters()):
                distance_loss += self.sigmoid(distance_criterion(paramA.detach(), paramB))

            distance_loss = -1 * distance_loss * self.dis_trade_off
            self.writer.add_scalars('data/loss', {
                'domain_cla2_loss': transfer_loss_2,
                'distance_loss': distance_loss,
                }, epoch)
            transfer_loss += distance_loss
        total_loss = transfer_loss + classifier_loss
        return total_loss


    def predict(self, inputs):
        features = self.f_net(inputs)
        features = self.bottleneck(features)
        _, softmax_outputs = self.c_net(features)
        return softmax_outputs

    #def get_parameter_list(self):
    #    return self.f_net.parameter_list + self.c_net.parameter_list + self.d_net_1.parameter_list + self.bottleneck.parameter_list

    def get_parameter_list(self):
        parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.bottleneck.parameters(), 'lr': 10},
                {'params': self.c_net.parameters(), 'lr': 10},
                {'params': self.d_net_1.parameters(), 'lr': 10},
                {'params': self.d_net_2.parameters(), 'lr': 10},
                ]
        return parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.bottleneck.train(mode)
        self.c_net.train(mode)
        self.d_net_1.train(mode)
        self.d_net_2.train(mode)
        self.is_train = mode
 
