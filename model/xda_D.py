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
    def __init__(self, feature_dim, hidden_dim=256):
        super(Discriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, 1)

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
        self.d_net = Discriminator(feature_dim=self.class_num)
        self.c_net_a = FeatureClassifier(feature_dim=self.high_feature_dim, class_num=class_num)
        self.c_net_b = FeatureClassifier(feature_dim=self.high_feature_dim, class_num=class_num)

        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.d_net = self.d_net.cuda()
            self.c_net_a = self.c_net_a.cuda()
            self.c_net_b = self.c_net_b.cuda()

    def get_loss(self, inputs, labels_source, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCELoss()

        high_features = self.f_net(inputs)
        src_high_features = high_features.narrow(0, 0, batch_size)
        tgt_high_features = high_features.narrow(0, batch_size, batch_size)

        src_outputs_a, src_probabilities_a = self.c_net_a(src_high_features)
        src_outputs_b, src_probabilities_b = self.c_net_b(src_high_features)
        tgt_outputs_a, tgt_probabilities_a = self.c_net_a(tgt_high_features)
        tgt_outputs_b, tgt_probabilities_b = self.c_net_b(tgt_high_features)

        #g_src_outputs_a, g_src_probabilities_a = self.c_net_a(src_high_features.detach())
        #g_src_outputs_b, g_src_probabilities_b = self.c_net_b(src_high_features.detach())
        #g_tgt_outputs_a, g_tgt_probabilities_a = self.c_net_a(tgt_high_features.detach())
        #g_tgt_outputs_b, g_tgt_probabilities_b = self.c_net_b(tgt_high_features.detach())
        #g_src = torch.abs(src_probabilities_a - src_probabilities_b)
        #g_tgt = torch.abs(tgt_probabilities_a - tgt_probabilities_b)
        #g_src = g_src_probabilities_a - g_src_probabilities_b
        #g_tgt = g_tgt_probabilities_a - g_tgt_probabilities_b

        g_src = torch.abs(src_probabilities_a - src_probabilities_b)
        g_tgt = torch.abs(tgt_probabilities_a - tgt_probabilities_b)
        g_inputs = torch.cat((g_src,g_tgt), dim=0)
        d_outputs = self.d_net(g_inputs)
        d_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
        if self.use_gpu:
            d_target = d_target.cuda()
        domain_loss = domain_criterion(d_outputs, d_target)

        classifier_loss_a = class_criterion(src_outputs_a, labels_source)
        classifier_loss_b = class_criterion(src_outputs_b, labels_source)
        classifier_loss = classifier_loss_a + classifier_loss_b
        self.writer.add_scalars('data/loss', {
            'classifier_loss_a': classifier_loss_a,
            'classifier_loss_b': classifier_loss_b,
            'domain_loss': domain_loss,
            }, epoch)
        self.writer.add_scalars('probabilities', {
            'src-a-max': torch.max(src_probabilities_a, 1)[0].mean(),
            'src-b-max': torch.max(src_probabilities_b, 1)[0].mean(),
            'tgt-a-max': torch.max(tgt_probabilities_a, 1)[0].mean(),
            'tgt-b-max': torch.max(tgt_probabilities_b, 1)[0].mean(),
            }, epoch)
        self.writer.add_scalars('data/distance', {
            'g_src': torch.mean(g_src.max(dim=1)[0]),
            'g_tgt': torch.mean(g_tgt.max(dim=1)[0]),
            }, epoch)
        return classifier_loss, -domain_loss, domain_loss

    def predict(self, inputs):
        tgt_high_features = self.f_net(inputs)
        _, softmax_outputs_a = self.c_net_a(tgt_high_features)
        _, softmax_outputs_b = self.c_net_b(tgt_high_features)
        softmax_outputs = (softmax_outputs_a + softmax_outputs_b) / 2
        return softmax_outputs

    def output(self, inputs):
        tgt_high_features = self.f_net(inputs)
        _, softmax_outputs_a = self.c_net_a(tgt_high_features)
        _, softmax_outputs_b = self.c_net_b(tgt_high_features)
        softmax_outputs = (softmax_outputs_a + softmax_outputs_b) / 2
        return tgt_high_features, softmax_outputs


    def get_parameter_list(self):
        classifier_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.c_net_a.parameters(), 'lr': 10},
                {'params': self.c_net_b.parameters(), 'lr': 10},
                ]
        max_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.c_net_a.parameters(), 'lr': 1},
                {'params': self.c_net_b.parameters(), 'lr': 1},
                ]
        min_parameter_list = [
                {'params': self.d_net.parameters(), 'lr': 10},
                ]

        return classifier_parameter_list, max_parameter_list, min_parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.c_net_a.train(mode)
        self.c_net_b.train(mode)
        self.d_net.train(mode)
        self.is_train = mode
 
