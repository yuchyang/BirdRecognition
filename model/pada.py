import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import model.backbone as backbone
from evaluator.predictor import predict_dataset

class PADAClassifier(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31):
        super(PADAClassifier, self).__init__()
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

class PADADiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(PADADiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.grl_layer = grl.GradientReverseLayer()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)

        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
    
        self.parameter_list = [{"params":self.ad_layer1.parameters(), "lr":10}, 
                            {"params":self.ad_layer2.parameters(), "lr":10}, 
                        {"params":self.ad_layer3.parameters(), "lr":10}]

    def forward(self, inputs):
        outputs = self.grl_layer(inputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs

class PADA(object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31, hidden_dim=1024, update_iter=500, trade_off=1.0, use_gpu=True):
        self.c_net = PADAClassifier(base_net, use_bottleneck, bottleneck_dim, class_num)

        if use_bottleneck:
            feature_dim = self.c_net.bottleneck_layer.out_features
        else:
            feature_dim = self.c_net.base_network.output_num()

        self.d_net = PADADiscriminator(feature_dim, hidden_dim)
        self.trade_off = trade_off
        self.use_gpu = use_gpu
        self.update_iter = update_iter
        self.is_train = False
        self.class_weight = torch.from_numpy(np.array([1.0] * class_num))

        if self.use_gpu:
            self.c_net = self.c_net.cuda()
            self.d_net = self.d_net.cuda()
            self.class_weight = self.class_weight.cuda()    

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss(weight = self.class_weight)
        labels_source_numpy = labels_source.data.cpu().numpy()
        batch_size = inputs.size(0) / 2
        dc_weight = torch.zeros(batch_size*2)
        for i in range(batch_size):
            dc_weight[i] = self.class_weight[int(labels_source_numpy[i])]
        dc_weight = dc_weight / torch.mean(dc_weight[0:batch_size])
        for i in range(batch_size, batch_size*2):
            dc_weight[i] = 1.0
        if self.use_gpu:
            dc_weight = dc_weight.cuda()
        transfer_criterion = nn.BCELoss(weight=dc_weight.view(-1))

        features, outputs, _ = self.c_net(inputs)
        dc_outputs = self.d_net(features)
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)
        dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + 
                    [[0]] * batch_size)).float())
        if self.use_gpu:
            dc_target = dc_target.cuda()
        transfer_loss = transfer_criterion(dc_outputs.view(-1), dc_target.view(-1))
        total_loss = self.trade_off * transfer_loss + classifier_loss
        return total_loss

    def update_class_weight(self, test_target_loader):
        ##TODO: change predict name : predict_dataset 
        ##TODO: can we use the same train_target_loader?
        ## may use test_target_loader first?
        outputs = predict_dataset(self, test_target_loader)
        self.class_weight = torch.mean(outputs, 0)
        self.class_weight = (self.class_weight / torch.max(self.class_weight)).view(-1) 

    #change to get_prediction?
    def predict(self, inputs):
        _, _, softmax_outputs = self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list + self.d_net.parameter_list

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
        self.d_net.train(mode)
        self.is_train = mode
 
