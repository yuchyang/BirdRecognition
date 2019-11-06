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


class BaggingDA(object):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0, use_gpu=False, writer=None):
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.writer = writer

        self.f_net = FeatureExtractor()
        high_feature_dim = self.f_net.output_dim()
        self.c_net = FeatureClassifier(feature_dim=high_feature_dim, class_num=class_num)
        self.c_net_a = FeatureClassifier(feature_dim=high_feature_dim, class_num=class_num)
        self.c_net_b = FeatureClassifier(feature_dim=high_feature_dim, class_num=class_num)

        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.c_net = self.c_net.cuda()
            self.c_net_a = self.c_net_a.cuda()
            self.c_net_b = self.c_net_b.cuda()

    def get_loss(self, inputs, labels, inputs_bags, labels_bags, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        distance_criterion = nn.L1Loss()

        high_features = self.f_net(inputs)
        src_high_features = high_features.narrow(0, 0, batch_size)
        tgt_high_features = high_features.narrow(0, batch_size, batch_size)

        # TODO CE on Bag influence the F or not
        bags_high_features = self.f_net(inputs_bags)
        a_high_features = bags_high_features.narrow(0, 0, batch_size)
        b_high_features = bags_high_features.narrow(0, batch_size, batch_size)
        a_labels = labels_bags.narrow(0, 0, batch_size)
        b_labels = labels_bags.narrow(0, batch_size, batch_size)

        src_outputs, src_probabilities = self.c_net(src_high_features)
        bag_outputs_a, bag_probabilities_a = self.c_net_a(a_high_features)
        bag_outputs_b, bag_probabilities_b = self.c_net_b(b_high_features)

        src_outputs_a, src_probabilities_a = self.c_net_a(src_high_features)
        src_outputs_b, src_probabilities_b = self.c_net_b(src_high_features)
        tgt_outputs_a, tgt_probabilities_a = self.c_net_a(tgt_high_features)
        tgt_outputs_b, tgt_probabilities_b = self.c_net_b(tgt_high_features)

        # classifier is trained by src
        classifier_loss = class_criterion(src_probabilities, labels)
        classifier_loss_a = class_criterion(bag_probabilities_a, a_labels)
        classifier_loss_b = class_criterion(bag_probabilities_b, b_labels)

        all_classifier_loss = classifier_loss + classifier_loss_a + classifier_loss_b

        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            'classifier_loss_a': classifier_loss_a,
            'classifier_loss_b': classifier_loss_b,
            }, epoch)
        self.writer.add_scalars('probabilities', {
            'src-a-max': torch.max(src_probabilities_a, 1)[0].mean(),
            'src-b-max': torch.max(src_probabilities_b, 1)[0].mean(),
            'tgt-a-max': torch.max(tgt_probabilities_a, 1)[0].mean(),
            'tgt-b-max': torch.max(tgt_probabilities_b, 1)[0].mean(),
            }, epoch)
        src_scale_loss = distance_criterion(src_probabilities_a, src_probabilities_b)
        tgt_scale_loss = distance_criterion(tgt_probabilities_a, tgt_probabilities_b)
        self.writer.add_scalars('data/distance', {
            'src_distance': src_scale_loss,
            'tgt_distance': tgt_scale_loss,
            }, epoch)
        return all_classifier_loss, self.alpha * torch.abs(tgt_scale_loss - src_scale_loss)

    def predict(self, inputs):
        tgt_high_features = self.f_net(inputs)
        #_, softmax_outputs = self.c_net(tgt_high_features)
        #return softmax_outputs
        _, a_softmax_outputs = self.c_net_a(tgt_high_features)
        _, b_softmax_outputs = self.c_net_b(tgt_high_features)
        return a_softmax_outputs + b_softmax_outputs

    def output(self, inputs):
        tgt_high_features = self.f_net(inputs)
        #_, softmax_outputs = self.c_net(tgt_high_features)
        #return tgt_high_features, softmax_outputs
        _, a_softmax_outputs = self.c_net_a(tgt_high_features)
        _, b_softmax_outputs = self.c_net_b(tgt_high_features)
        return tgt_high_features, a_softmax_outputs + b_softmax_outputs

    def get_parameter_list(self):
        classifier_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.c_net.parameters(), 'lr': 10},
                {'params': self.c_net_a.parameters(), 'lr': 10},
                {'params': self.c_net_b.parameters(), 'lr': 10},
                ]
        scale_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                ]

        return classifier_parameter_list, scale_parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.c_net.train(mode)
        self.c_net_a.train(mode)
        self.c_net_b.train(mode)
        self.is_train = mode
 
