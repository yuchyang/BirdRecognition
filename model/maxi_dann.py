import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
from loss import loss
import model.backbone as backbone

class DANNClassifier(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31):
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
        softmax_outputs = self.softmax(outputs)
        return features, outputs, softmax_outputs

class DANNDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_size):
        super(DANNDiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, output_size)
        self.ad_layer4 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.grl_layer = grl.GradientReverseLayer()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)

        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.01)
        self.ad_layer4.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.ad_layer4.bias.data.fill_(0.0)
    
        self.parameter_list = [{"params":self.ad_layer1.parameters(), "lr":10}, 
                            {"params":self.ad_layer2.parameters(), "lr":10}, 
                        {"params":self.ad_layer3.parameters(), "lr":10},
                        {"params":self.ad_layer4.parameters(), "lr":10}]

    def forward(self, inputs):
        outputs = self.grl_layer(inputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        #outputs = self.ad_layer4(feature_outputs)

        dis_outputs = inputs.detach()
        dis_outputs = self.drop_layer1(self.relu(self.ad_layer1(dis_outputs)))
        dis_outputs = self.drop_layer2(self.relu(self.ad_layer2(dis_outputs)))
        dis_feature_outputs = self.drop_layer3(self.relu(self.ad_layer4(dis_outputs)))

        return dis_feature_outputs, outputs

class DANN(object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31, hidden_dim=1024, output_size=16, trade_off=1.0, dis_trade_off=1.0, use_gpu=True, writer=None):
        self.c_net = DANNClassifier(base_net, use_bottleneck, bottleneck_dim, class_num)

        if use_bottleneck:
            feature_dim = self.c_net.bottleneck_layer.out_features
        else:
            feature_dim = self.c_net.base_network.output_num()

        self.writer = writer
        self.d_net_1 = DANNDiscriminator(feature_dim, hidden_dim, output_size)
        self.d_net_2 = DANNDiscriminator(feature_dim, hidden_dim, output_size)
        self.trade_off = trade_off
        self.dis_trade_off = dis_trade_off
        self.use_gpu = use_gpu
        self.is_train = False

        if self.use_gpu:
            self.c_net = self.c_net.cuda()
            self.d_net_1 = self.d_net_1.cuda()
            self.d_net_2 = self.d_net_2.cuda()
    
    def get_loss(self, inputs, labels_source, epoch):
        batch_size = inputs.size(0) // 2
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        #transfer_criterion = nn.BCEWithLogitsLoss()
        #distance_criterion = loss.mmd_loss
        distance_criterion = nn.L1Loss()
        features, outputs, _ = self.c_net(inputs)
        dc_features_1, dc_outputs_1 = self.d_net_1(features)
        dc_features_1, dc_outputs_2 = self.d_net_2(features)
        #dc_features_1, _ = self.d_net_1(features.detach())
        #dc_features_2, _ = self.d_net_2(features.detach())
        classifier_loss = class_criterion(outputs.narrow(0, 0, batch_size), labels_source)
        dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + 
                    [[0]] * batch_size)).float())
        if self.use_gpu:
            dc_target = dc_target.cuda()
        transfer_loss = transfer_criterion(dc_outputs_1, dc_target) + transfer_criterion(dc_outputs_2, dc_target)
        #mmd_loss = -distance_criterion(dc_features_1, dc_features_2, 2.0, 5, None)
        mmd_loss = -distance_criterion(dc_features_1, dc_features_2)

        total_loss = self.trade_off * transfer_loss + classifier_loss + mmd_loss * self.dis_trade_off
        self.writer.add_scalars('data/loss', {
            'mmd_loss': mmd_loss,
            'domain_classification_loss': transfer_loss,
            'classifier_loss': classifier_loss,
            }, epoch)

        return total_loss

    def predict(self, inputs):
        _, _, softmax_outputs = self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list + self.d_net_1.parameter_list + self.d_net_2.parameter_list

    def save_model(self, c_net_path=None, d_net_path=None):
        if c_net_path:
            torch.save(self.c_net.state_dict(), c_net_path)
        if d_net_path:
            torch.save(self.d_net.state_dict(), d_net_path)

    def load_model(self, c_net_path=None, d_net_path=None):
        if c_net_path:
            self.c_net.load_state_dict(torch.load(c_net_path))
        if d_net_path:
            self.d_net.load_state_dict(torch.load(d_net_path))

    def set_train(self, mode):
        self.c_net.train(mode)
        self.d_net_1.train(mode)
        self.d_net_2.train(mode)
        self.is_train = mode
 
