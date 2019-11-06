import torch
import torch.nn as nn
import loss
from model import backbone
from torchvision import models
from layer.grl import GradientReverseLayer, AdaptorLayer
from layer.silence import SilenceLayer
from layer.rmm import RMMLayer
from layer import grl
from loss import loss
import copy
from easy import BCELossForMultiClassification
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.nn.modules.batchnorm import BatchNorm1d
import torch.nn.functional as F
import numpy as np
import random


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
        x = self.avgpool(x)
        high_features = x.view(x.size(0), -1)
        return high_features

    def output_dim(self):
        return self.high_dim


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=31):
        super(FeatureClassifier, self).__init__()
        self.classifier_layer_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.classifier_layer_2 = nn.Linear(feature_dim // 2, class_num)
        self.bn = BatchNorm1d(feature_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

        ## initialization
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        self.classifier_layer_1.weight.data.normal_(0, 0.01)
        self.classifier_layer_1.bias.data.fill_(0.0)
        self.classifier_layer_2.weight.data.normal_(0, 0.01)
        self.classifier_layer_2.bias.data.fill_(0.0)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.dropout(self.relu(self.bn(self.classifier_layer_1(outputs))))
        outputs = self.classifier_layer_2(outputs)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs


class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, feature_dim // 2)
        self.ad_layer2 = nn.Linear(feature_dim // 2, feature_dim // 2)
        self.ad_layer3 = nn.Linear(feature_dim // 2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)

        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
    
    def forward(self, inputs):
        outputs = inputs
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class XDA(object):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0, use_gpu=False, writer=None):
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.writer = writer
        self.class_num = class_num

        self.f_net = FeatureExtractor()
        self.high_feature_dim = self.f_net.output_dim()
        self.c_net_a = FeatureClassifier(feature_dim=self.high_feature_dim, class_num=class_num)
        self.src_centor = torch.zeros([self.class_num, self.class_num])
        self.src_counter = torch.ones([self.class_num])
        # centor for each class

        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.c_net_a = self.c_net_a.cuda()
            self.src_centor = self.src_centor.cuda()
            self.src_counter = self.src_counter.cuda()

    def get_loss(self, inputs, labels_source, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        distance_criterion = nn.L1Loss()

        high_feature_dim = self.f_net.output_dim()
        c_net_b = FeatureClassifier(feature_dim=self.high_feature_dim, class_num=self.class_num)
        if self.use_gpu:
            c_net_b = c_net_b.cuda()

        high_features = self.f_net(inputs)
        src_high_features = high_features.narrow(0, 0, batch_size)
        tgt_high_features = high_features.narrow(0, batch_size, batch_size)

        src_outputs_a, src_probabilities_a = self.c_net_a(src_high_features)
        tgt_outputs_a, tgt_probabilities_a = self.c_net_a(tgt_high_features)
        src_selection = torch.argmax(src_probabilities_a, dim=1).cuda()
        tgt_selection = torch.argmax(tgt_probabilities_a, dim=1).cuda()
        tgt_centor = torch.zeros([self.class_num, self.class_num]).cuda()
        tgt_counter = torch.ones([self.class_num]).cuda()
        for b_i in range(batch_size):
            self.src_centor[src_selection[b_i]] += src_probabilities_a[b_i].detach()
            self.src_counter[src_selection[b_i]] += 1
            tgt_centor[tgt_selection[b_i]] += tgt_probabilities_a[b_i].detach()
            tgt_counter[tgt_selection[b_i]] += 1
        src_curr_centor = torch.transpose(self.src_centor.transpose(0,1).detach() / self.src_counter.float(), 0, 1)
        tgt_curr_centor = torch.transpose(tgt_centor.transpose(0,1) / tgt_counter.float(), 0, 1)
        centor_loss = distance_criterion(src_curr_centor, tgt_curr_centor)# / (self.class_num * self.class_num)

        src_outputs_b, src_probabilities_b = c_net_b(src_high_features)
        tgt_outputs_b, tgt_probabilities_b = c_net_b(tgt_high_features)
        classifier_loss = class_criterion(src_outputs_a, labels_source)
        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            'centor_loss': centor_loss,
            }, epoch)
        self.writer.add_scalars('probabilities', {
            'src-a-max': torch.max(src_probabilities_a, 1)[0].mean(),
            'src-b-max': torch.max(src_probabilities_b, 1)[0].mean(),
            'tgt-a-max': torch.max(tgt_probabilities_a, 1)[0].mean(),
            'tgt-b-max': torch.max(tgt_probabilities_b, 1)[0].mean(),
            }, epoch)
        src_scale_loss = distance_criterion(src_probabilities_a, src_probabilities_b) / (self.class_num * batch_size)
        tgt_scale_loss = distance_criterion(tgt_probabilities_a, tgt_probabilities_b) / (self.class_num * batch_size)
        self.writer.add_scalars('data/distance', {
            'src_distance': src_scale_loss,
            'tgt_distance': tgt_scale_loss,
            }, epoch)
        return classifier_loss, torch.abs(tgt_scale_loss - src_scale_loss) + self.beta * centor_loss

    def get_centor_loss(self, inputs_list, epoch):
        batch_size = inputs_list[0].size(0) // 2
        distance_criterion = nn.L1Loss()

        src_centor = torch.zeros([self.class_num, self.class_num]).cuda()
        tgt_centor = torch.zeros([self.class_num, self.class_num]).cuda()
        src_class_counter = torch.ones([self.class_num]).cuda()
        tgt_class_counter = torch.ones([self.class_num]).cuda()
        for inputs in inputs_list:
            high_features = self.f_net(inputs)
            src_high_features = high_features.narrow(0, 0, batch_size)
            tgt_high_features = high_features.narrow(0, batch_size, batch_size)
    
            src_outputs_a, src_probabilities_a = self.c_net_a(src_high_features)
            tgt_outputs_a, tgt_probabilities_a = self.c_net_a(tgt_high_features)
    
            src_selection = torch.argmax(src_probabilities_a, dim=1)
            tgt_selection = torch.argmax(tgt_probabilities_a, dim=1)
            for b_i in range(batch_size):
                src_centor[src_selection[b_i]] += src_probabilities_a[b_i]
                src_class_counter[src_selection[b_i]] += 1
                tgt_centor[tgt_selection[b_i]] += tgt_probabilities_a[b_i]
                tgt_class_counter[tgt_selection[b_i]] += 1

        src_curr_centor = torch.transpose(src_centor.transpose(0,1) / src_class_counter.float(), 0 ,1)
        tgt_curr_centor = torch.transpose(tgt_centor.transpose(0,1) / tgt_class_counter.float(), 0 ,1)
        centor_loss = distance_criterion(src_curr_centor, tgt_curr_centor) / (self.class_num * self.class_num)
    
        self.writer.add_scalars('data/loss', {
            'centor_loss': centor_loss,
            }, epoch)
        return centor_loss


    def predict(self, inputs):
        tgt_high_features = self.f_net(inputs)
        _, softmax_outputs_a = self.c_net_a(tgt_high_features)
        softmax_outputs = softmax_outputs_a
        return softmax_outputs

    def output(self, inputs):
        tgt_high_features = self.f_net(inputs)
        _, softmax_outputs_a = self.c_net_a(tgt_high_features)
        softmax_outputs = softmax_outputs_a
        return tgt_high_features, softmax_outputs


    def get_parameter_list(self):
        classifier_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.c_net_a.parameters(), 'lr': 10},
                ]
        scale_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                ]

        return classifier_parameter_list, scale_parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.c_net_a.train(mode)
        self.is_train = mode
 
