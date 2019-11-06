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
from easy import BCELossForMultiClassification
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
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


class DomainAdaptor(nn.Module):
    def __init__(self, feature_dim=2048, adaptor_dim=256):
        super(DomainAdaptor, self).__init__()
        ## initialization
        self.bottleneck_layer = spectral_norm(nn.Linear(feature_dim, adaptor_dim))
        self.bottleneck_layer.weight.data.normal_(0, 0.005)
        self.bottleneck_layer.bias.data.fill_(0.1)

    def forward(self, inputs):
        adapted_features = self.bottleneck_layer(inputs)
        return adapted_features


class Adjustor(nn.Module):
    def __init__(self, forward_rate=1.0, backward_rate=1.0):
        super(Adjustor, self).__init__()
        ## initialization
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def forward(self, inputs):
        adaptor = AdaptorLayer(forward_rate=self.forward_rate, backward_rate=self.backward_rate)
        outputs = adaptor(inputs)
        return outputs


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


class CONDAN(object):
    def __init__(self, base_net='ResNet50', class_num=31, use_bottleneck=True, adaptor_dim=256, hidden_dim=1024, trade_off=1.0, use_gpu=False, writer=None):
        self.use_gpu = use_gpu
        self.trade_off = trade_off
        self.writer = writer

        self.f_net = FeatureExtractor()
        high_feature_dim = self.f_net.output_dim()
        self.src_adaptor = DomainAdaptor(feature_dim=high_feature_dim, adaptor_dim=adaptor_dim)
        self.tgt_adaptor = DomainAdaptor(feature_dim=high_feature_dim, adaptor_dim=adaptor_dim)
        self.src_adjustor = Adjustor(forward_rate=0.9, backward_rate=0)
        self.tgt_adjustor = Adjustor(forward_rate=0.1, backward_rate=0.1)
        #feature_dim = adaptor_dim
        self.c_net = FeatureClassifier(feature_dim=adaptor_dim, class_num=class_num)

        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.src_adaptor = self.src_adaptor.cuda()
            self.tgt_adaptor = self.tgt_adaptor.cuda()
            self.src_adjustor = self.src_adjustor.cuda()
            self.tgt_adjustor = self.tgt_adjustor.cuda()
            self.c_net = self.c_net.cuda()

    def get_loss(self, inputs, labels_source, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        sigmoid = nn.Sigmoid()

        high_features  = self.f_net(inputs)
        src_high_features = high_features.narrow(0, 0, batch_size)
        tgt_high_features = high_features.narrow(0, batch_size, batch_size)
        src_high_features = self.src_adaptor(src_high_features)
        tgt_high_features_a = self.src_adaptor(tgt_high_features)
        tgt_high_features_a = self.src_adjustor(tgt_high_features_a)
        tgt_high_features_b = self.tgt_adaptor(tgt_high_features)
        tgt_high_features_b = self.tgt_adjustor(tgt_high_features_b)
        tgt_high_features = tgt_high_features_a + tgt_high_features_b

        src_outputs, src_probabilities = self.c_net(src_high_features)
        tgt_outputs, tgt_probabilities = self.c_net(tgt_high_features)
        classifier_loss = class_criterion(src_outputs, labels_source)
        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            }, epoch)
        src_entropy_loss = loss.entropy_loss_digits(src_outputs) / batch_size
        tgt_entropy_loss = loss.entropy_loss_digits(tgt_outputs) / batch_size
        self.writer.add_scalars('probabilities', {
            'src-max': torch.max(src_probabilities, 1)[0].mean(),
            'tgt-max': torch.max(tgt_probabilities, 1)[0].mean(),
            }, epoch)
        self.writer.add_scalars('data/loss', {
            'src_entropy': src_entropy_loss,
            'tgt_entropy': tgt_entropy_loss,
            }, epoch)

        src_l2_reg = None
        for W in self.src_adaptor.parameters():
            if src_l2_reg is None:
                src_l2_reg = W.norm(2)
            else:
                src_l2_reg = src_l2_reg + W.norm(2)
        tgt_l2_reg = None
        for W in self.tgt_adaptor.parameters():
            if tgt_l2_reg is None:
                tgt_l2_reg = W.norm(2)
            else:
                tgt_l2_reg = tgt_l2_reg + W.norm(2)
        src_loss = classifier_loss
        src_entropy_loss = -1 * self.trade_off * src_entropy_loss + src_l2_reg * 0.2
        tgt_entropy_loss = self.trade_off * tgt_entropy_loss + tgt_l2_reg * 0.2
        return src_loss, src_entropy_loss, tgt_entropy_loss

    def predict(self, inputs):
        tgt_high_features = self.f_net(inputs)
        # features = self.bottleneck(high_features)
        tgt_high_features_a = self.src_adaptor(tgt_high_features)
        tgt_high_features_a = self.src_adjustor(tgt_high_features_a)
        tgt_high_features_b = self.tgt_adaptor(tgt_high_features)
        tgt_high_features_b = self.tgt_adjustor(tgt_high_features_b)
        tgt_high_features = tgt_high_features_a + tgt_high_features_b
        _, softmax_outputs = self.c_net(tgt_high_features)
        return softmax_outputs

    def get_parameter_list(self):
        src_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.src_adaptor.parameters(), 'lr': 10},
                {'params': self.c_net.parameters(), 'lr': 10},
                ]
        src_entropy_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.src_adaptor.parameters(), 'lr': 10},
                ]
        tgt_entropy_parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.src_adaptor.parameters(), 'lr': 10},
                {'params': self.tgt_adaptor.parameters(), 'lr': 10},
                {'params': self.src_adjustor.parameters(), 'lr': 10},
                {'params': self.tgt_adjustor.parameters(), 'lr': 10},
                ]

        return src_parameter_list, src_entropy_parameter_list, tgt_entropy_parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.c_net.train(mode)
        self.src_adaptor.train(mode)
        self.tgt_adaptor.train(mode)
        self.src_adjustor.train(mode)
        self.tgt_adjustor.train(mode)
        self.is_train = mode
 
