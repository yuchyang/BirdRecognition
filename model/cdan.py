import torch
import torch.nn as nn
import loss
from model import backbone
from torchvision import models
from layer.grl import GradientReverseLayer
from layer.silence import SilenceLayer
from layer.rmm import RMMLayer
from layer import grl
from loss import loss
from easy import BCELossForMultiClassification
from torch.autograd import Variable
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


class CDANDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, randomize=False):
        super(CDANDiscriminator, self).__init__()
        self.randomize = randomize

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)
        self.rmm_layer = RMMLayer()

        self.forward_count = 0

    def forward(self, features, softmax_outputs):
        self.forward_count += 1
        grl_layer = grl.GradientReverseLayer(iter_num=self.forward_count)

        if self.randomize:
            rman_out_list = self.rmm_layer.forward([features, softmax_outputs])
            rman_out = rman_out_list[0]
            for rman_single_out in rman_out_list[1:]:
                rman_out = torch.mul(rman_out, rman_single_out)
            adv_inputs = rman_out.view(-1, rman_out.size(1))
        else:
            op_out = torch.bmm(softmax_outputs.unsqueeze(2), features.unsqueeze(1))
            adv_inputs = op_out.view(-1, softmax_outputs.size(1) * features.size(1))
        adv_inputs = torch.sqrt(torch.abs(adv_inputs) + 1e-12)
        adv_inputs = F.normalize(adv_inputs, p=2, dim=1)
        outputs = grl_layer(adv_inputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs


class CDAN(object):
    def __init__(self, base_net='ResNet50', class_num=31, randomize=False, use_bottleneck=True, bottleneck_dim=256, hidden_dim=1024, trade_off=1.0, use_gpu=False, writer=None):
        self.randomize = randomize
        self.use_gpu = use_gpu
        self.trade_off = trade_off
        self.writer = writer

        self.f_net = FeatureExtractor()
        high_feature_dim = self.f_net.output_dim()
        self.bottleneck = Bottleneck(feature_dim=high_feature_dim, bottleneck_dim=bottleneck_dim)
        feature_dim = bottleneck_dim
        self.c_net = FeatureClassifier(feature_dim=feature_dim, class_num=class_num)
        self.discriminator = CDANDiscriminator(feature_dim * class_num, hidden_dim, randomize)

        if self.use_gpu:
            self.f_net = self.f_net.cuda()
            self.bottleneck = self.bottleneck.cuda()
            self.c_net = self.c_net.cuda()
            self.discriminator = self.discriminator.cuda()

    def get_loss(self, inputs, labels_source, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()

        high_features  = self.f_net(inputs)
        high_features = self.bottleneck(high_features)
        outputs, probabilities = self.c_net(high_features)
        classifier_loss = class_criterion(outputs.narrow(0, 0, batch_size), labels_source)
        self.writer.add_scalars('data/loss', {
            'classifier_loss': classifier_loss,
            }, epoch)

        dc_output = self.discriminator(high_features, probabilities)
        dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
        if self.use_gpu:
            dc_target = dc_target.cuda()
        # transfer_loss = transfer_criterion(dc_output, dc_target)
        # Try exp(H(g))
        dc_weights = torch.exp(probabilities)
        transfer_loss = BCELossForMultiClassification(dc_target, dc_output, instance_level_weight=dc_weights).cuda()

        dc_output_src = dc_output.narrow(0, 0, batch_size)
        dc_output_tgt = dc_output.narrow(0, batch_size, batch_size)
        self.writer.add_scalars('probabilities/domainclassifier', {
            'src': dc_output_src.mean(),
            'tgt': dc_output_tgt.mean(),
            }, epoch)
        self.writer.add_scalars('data/loss', {
            'transfer_loss': transfer_loss,
            }, epoch)

        total_loss = self.trade_off * transfer_loss + classifier_loss
        return total_loss

    def predict(self, inputs):
        high_features = self.f_net(inputs)
        features = self.bottleneck(high_features)
        _, softmax_outputs = self.c_net(features)
        return softmax_outputs

    def get_parameter_list(self):
        parameter_list = [
                {'params': self.f_net.parameters(), 'lr': 1},
                {'params': self.bottleneck.parameters(), 'lr': 10},
                {'params': self.c_net.parameters(), 'lr': 10},
                {'params': self.discriminator.parameters(), 'lr': 10},
                ]
        return parameter_list

    def set_train(self, mode):
        self.f_net.train(mode)
        self.bottleneck.train(mode)
        self.c_net.train(mode)
        self.discriminator.train(mode)
        self.is_train = mode
 
