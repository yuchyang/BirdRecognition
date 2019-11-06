import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import model.backbone as backbone
from tensorboardX import SummaryWriter
from loss.loss import entropy_loss_digits
import os

class Feature(nn.Module):
    def __init__(self, base_net='ResNet50', lr_ratio=1):
        super(Feature, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        ## collect parameters
        # TODO Attention !! parameter_list CANNOT be here, to make Function save() work without generator error
        #self.parameter_list = [
        #        {"params":self.base_network.parameters(), "lr":1}
        #        ]
    def forward(self, inputs):
        features = self.base_network(inputs)
        return features

    def output_dim(self):
        return self.base_network.output_num()

class DomainClassifier(nn.Module):
    def __init__(self, base_net='ResNet50', bottleneck_dim=256, class_num=2, lr_ratio=1):
        super(DomainClassifier, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.classifier_layer = nn.Linear(self.base_network.output_num(), class_num)
        self.softmax = nn.Softmax()

        ## initialization
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

        ## collect parameters
        #self.parameter_list = [
        #        {"params":self.base_network.parameters(), "lr":lr_ratio}, 
        #        {"params":self.bottleneck_layer.parameters(), "lr":10*lr_ratio}, 
        #        {"params":self.classifier_layer.parameters(), "lr":10*lr_ratio}
        #        ]

    def forward(self, inputs):
        features = self.base_network(inputs)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=31, lr_ratio=10):
        super(FeatureClassifier, self).__init__()
        ## set base network
        self.classifier_layer = nn.Linear(256, class_num)
        self.softmax = nn.Softmax()

        ## initialization
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

        ## collect parameters
        #self.parameter_list = [{"params":self.classifier_layer.parameters(), "lr":lr_ratio}]
         
    def forward(self, inputs):
        outputs = self.classifier_layer(inputs)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs

class FeatureDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lr_ratio=10):
        super(FeatureDiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim,hidden_dim)
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
        self.iter_count = 0
        
        #self.parameter_list = [{"params":self.ad_layer1.parameters(), "lr":lr_ratio}, 
        #                    {"params":self.ad_layer2.parameters(), "lr":lr_ratio}, 
        #                {"params":self.ad_layer3.parameters(), "lr":lr_ratio}]

    def forward(self, inputs):
        self.iter_count += 1
        grl_layer = grl.GradientReverseLayer(iter_num=self.iter_count)
        outputs = grl_layer(inputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs

class MSDA(nn.Module):
    def __init__(self, base_net='ResNet50', bottleneck_dim=256, class_num=31, hidden_dim=1024, feature_adv_tradeoff=1.0, domain_adv_tradeoff=2.0, consistency_tradeoff=0.2, dc_entropy_tradeoff=1.0, use_gpu=True):
        super(MSDA, self).__init__()

        self.base_net = base_net
        self.bottleneck_dim = bottleneck_dim
        self.class_num = class_num
        self.hidden_dim = hidden_dim
        self.feature_adv_tradeoff = feature_adv_tradeoff
        self.domain_adv_tradeoff = domain_adv_tradeoff
        self.consistency_tradeoff = consistency_tradeoff
        self.dc_entropy_tradeoff = dc_entropy_tradeoff
        self.use_gpu = use_gpu

        self.SharedFNet = Feature(base_net=base_net)
        self.F1 = nn.Linear(self.SharedFNet.output_dim(), bottleneck_dim)
        self.F2 = nn.Linear(self.SharedFNet.output_dim(), bottleneck_dim)
        self.D0 = DomainClassifier(base_net='ResNet50', bottleneck_dim=256, class_num=2)
        self.D1 = FeatureDiscriminator(feature_dim=bottleneck_dim, hidden_dim=hidden_dim)
        self.D2 = FeatureDiscriminator(feature_dim=bottleneck_dim, hidden_dim=hidden_dim)
        self.C1 = FeatureClassifier(feature_dim=bottleneck_dim, class_num=class_num)
        self.C2 = FeatureClassifier(feature_dim=bottleneck_dim, class_num=class_num)

        self.class_criterion = nn.CrossEntropyLoss()

        # TODO SummaryWriter CANNOT be here to make save() work without thread error
        #self.writer = SummaryWriter() 

        # initialization
        self.F1.weight.data.normal_(0, 0.01)
        self.F1.bias.data.fill_(0.0)
        self.F2.weight.data.normal_(0, 0.01)
        self.F2.bias.data.fill_(0.0)

        if self.use_gpu:
            self.SharedFNet = self.SharedFNet.cuda()
            self.F1 = self.F1.cuda()
            self.F2 = self.F2.cuda()
            self.D0 = self.D0.cuda()
            self.D1 = self.D1.cuda()
            self.D2 = self.D2.cuda()
            self.C1 = self.C1.cuda()
            self.C2 = self.C2.cuda()

    def get_parameter_list(self):
        parameter_list = [
                {"params":self.SharedFNet.parameters(), "lr":1},
                {"params":self.F1.parameters(), "lr":10},
                {"params":self.F2.parameters(), "lr":10},
                {"params":self.D0.parameters(), "lr":1},
                {"params":self.D1.parameters(), "lr":10},
                {"params":self.D2.parameters(), "lr":10},
                {"params":self.C1.parameters(), "lr":10},
                {"params":self.C2.parameters(), "lr":10},
                ]
        return parameter_list

    def multiply(self, x, y):
        return torch.transpose(x * torch.transpose(y, 0, 1), 0, 1)

    def get_loss(self, inputs_sa, inputs_sb, labels_sa, labels_sb, inputs_t, epoch, writer=None):
        batch_size = inputs_sa.size(0)

        combined_inputs_ab = torch.cat((inputs_sa, inputs_sb), 0)
        combined_dc_outputs, combined_dc_probabilities = self.D0(combined_inputs_ab)
        dc_outputs_sa, dc_outputs_sb = torch.split(combined_dc_outputs, [batch_size, batch_size], 0)
        dc_probabilities_sa, dc_probabilities_sb = torch.split(combined_dc_probabilities, [batch_size, batch_size], 0)
        combined_inputs_abt = torch.cat((inputs_sa, inputs_sb, inputs_t), 0)
        combined_features = self.SharedFNet(combined_inputs_abt)
        features_sa, features_sb, features_t = torch.split(combined_features, [batch_size, batch_size, batch_size], 0)
        features_sa = self.F1(features_sa)
        features_sb = self.F2(features_sb)
        features_at = self.F1(features_t)
        features_bt = self.F2(features_t)
        class_outputs_sa, class_probabilities_sa = self.C1(features_sa)
        class_outputs_sb, class_probabilities_sb = self.C2(features_sb)
        class_outputs_at, class_probabilities_at = self.C1(features_at)
        class_outputs_bt, class_probabilities_bt = self.C2(features_bt)
        adv_outputs_sa = self.D1(features_sa)
        adv_outputs_at = self.D1(features_at)
        adv_outputs_sb = self.D2(features_sb)
        adv_outputs_bt = self.D2(features_bt)
        _, weights = self.D0(inputs_t)
        #W1 = weights.detach()[:,0]
        #W2 = weights.detach()[:,1]
        W1 = torch.zeros(weights.detach()[:,0].shape, dtype=torch.float32).cuda()
        W2 = torch.ones(weights.detach()[:,1].shape, dtype=torch.float32).cuda()


        dc_target_sa = Variable(torch.from_numpy(np.array([0] * batch_size)).long())
        dc_target_sb = Variable(torch.from_numpy(np.array([1] * batch_size)).long())
        adv_target_s = Variable(torch.from_numpy(np.array([1] * batch_size)).float())
        adv_target_t = Variable(torch.from_numpy(np.array([0] * batch_size)).float())
        if self.use_gpu:
            dc_target_sa = dc_target_sa.cuda()
            dc_target_sb = dc_target_sb.cuda()
            adv_target_s = adv_target_s.cuda()
            adv_target_t = adv_target_t.cuda()
        # check the accuracy of Domain Classifier
        _, d_predict_sa = torch.max(dc_probabilities_sa, 1)
        _, d_predict_sb = torch.max(dc_probabilities_sb, 1)
        dc_accuracy_sa = float(torch.sum(torch.squeeze(d_predict_sa).long() == dc_target_sa)) / float(dc_target_sa.size()[0])
        dc_accuracy_sb = float(torch.sum(torch.squeeze(d_predict_sb).long() == dc_target_sb)) / float(dc_target_sb.size()[0])

        dc_entropy_loss = entropy_loss_digits(combined_dc_probabilities) / batch_size
        classifier_loss_sa = self.class_criterion(class_outputs_sa, labels_sa)
        classifier_loss_sb = self.class_criterion(class_outputs_sb, labels_sb)
        classifier_loss = classifier_loss_sa + classifier_loss_sb
        dc_classifier_loss_sa = self.class_criterion(dc_outputs_sa, dc_target_sa)
        dc_classifier_loss_sb = self.class_criterion(dc_outputs_sb, dc_target_sb)
        dc_classifier_loss = dc_classifier_loss_sa + dc_classifier_loss_sb
        transfer_loss_at = nn.BCELoss(weight=W1)(adv_outputs_sa, adv_target_s) + nn.BCELoss(weight=W1)(adv_outputs_at, adv_target_t)
        transfer_loss_bt = nn.BCELoss(weight=W2)(adv_outputs_sb, adv_target_s) + nn.BCELoss(weight=W2)(adv_outputs_bt, adv_target_t)
        transfer_loss = transfer_loss_at + transfer_loss_bt
        probabilities = self.multiply(W1, class_probabilities_at) + self.multiply(W2, class_probabilities_bt)
        consistency_loss = entropy_loss_digits(probabilities) / batch_size

        total_loss = classifier_loss + self.domain_adv_tradeoff * dc_classifier_loss + self.feature_adv_tradeoff * transfer_loss + self.consistency_tradeoff * consistency_loss + self.dc_entropy_tradeoff * dc_entropy_loss 

        if writer != None:
            writer.add_scalars('data/domain_classifier', {
                'sa-accu': dc_accuracy_sa,
                'sb-accu': dc_accuracy_sb,
                'accu': (dc_accuracy_sa + dc_accuracy_sb)/2
                }, epoch)
            writer.add_scalars('data/weights', {
                'W1': W1.mean(),
                'W2': W2.mean(),
                }, epoch)
            writer.add_scalars('data/losses', {
                'classifier_loss': classifier_loss,
                'dc_classifier_loss': dc_classifier_loss,
                'transfer_loss': transfer_loss,
                'consistency_loss': consistency_loss,
                'total_loss': total_loss,
                }, epoch)
        return total_loss

    def predict(self, inputs):
        features_t = self.SharedFNet(inputs)
        features_at = self.F2(features_t)
        features_bt = self.F2(features_t)
        class_outputs_at, class_probabilities_at = self.C1(features_at)
        class_outputs_bt, class_probabilities_bt = self.C2(features_bt)
        _, weights = self.D0(inputs)
        #W1 = weights.detach()[:,0]
        #W2 = weights.detach()[:,1]
        W1 = torch.zeros(weights.detach()[:,0].shape, dtype=torch.float32).cuda()
        W2 = torch.ones(weights.detach()[:,1].shape, dtype=torch.float32).cuda()
        probabilities = self.multiply(W1, class_probabilities_at) + self.multiply(W2, class_probabilities_bt)
        #min_vec = torch.min(probabilities, 1)
        #max_vec = torch.max(probabilities, 1)
        #normalized_probabilities = torch.div(torch.sub(probabilities, min_vec), torch.sub(max_vec, min_vec))
        return probabilities

    def set_train(self, mode):
        self.SharedFNet.train(mode)
        self.F1.train(mode)
        self.F2.train(mode)
        self.D0.train(mode)
        self.D1.train(mode)
        self.D2.train(mode)
        self.C1.train(mode)
        self.C2.train(mode)
        self.is_train = mode
 
