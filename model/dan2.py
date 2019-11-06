import torch
import torch.nn as nn
import layer
import loss
import network

import numpy as np
import torch.utils.data as util_data
import lr_schedule
from torch.autograd import Variable

class DAN_Net(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31):
        super(DAN_Net, self).__init__()

        ## set base network
        ##TODO:How to process pretrained model? Put network.py as pretrained.py         in model directory?
        self.base_network = network.network_dict[base_net]()
        if use_bottleneck:
            bottleneck_layer = nn.Linear(self.base_network.output_num(), bottleneck_dim)
            self.classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
        else:
            self.classifier_layer = nn.Linear(self.base_network.output_num(), class_num)

        ## initialization
        if use_bottleneck:
            bottleneck_layer.weight.data.normal_(0, 0.005)
            bottleneck_layer.bias.data.fill_(0.1)
            self.bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if use_bottleneck:
                self.bottleneck_layer = self.bottleneck_layer.cuda()
            self.classifier_layer = self.classifier_layer.cuda()
            self.base_network = base_network.cuda()

        ## collect parameters
        if use_bottleneck:
            self.parameter_list = [{"params":self.base_network.parameters(), "lr":1}, {"params":self.bottleneck_layer.parameters(), "lr":10}, {"params":self.classifier_layer.parameters(), "lr":10}]
       
        else:
            self.parameter_list = [{"params":self.base_network.parameters(), "lr":1}, {"params":self.classifier_layer.parameters(), "lr":10}]

                        
    def forward(self, inputs):
        features = self.base_network(inputs)
        if use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        return features, outputs


class DAN(Object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31):
        self.net = DAN_Net(base_net, use_bottleneck, bottleneck_dim, class_num)

    def train_network(source_loader, target_loader, test_loader, optimizer, num_iterations=20000, test_interval=500, loss_config):
        len_train_source = len(source_loader) - 1
        len_train_target = len(target_loader) - 1
        best_acc = 0.0

        for i in range(num_iterations):
            ## test in the train
            if i % test_interval == 0:
                self.net.train(False)
                ## TODO:evaluate code, can use the same

            self.net.train(True)

            ## TODO:where to initialize optimizer, also need params
            
            optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
            optimizer.zero_grad()

            if i % len_train_source == 0:
                iter_source = iter(source_loader)
            if i % len_train_target == 0:
                iter_target = iter(target_loader)
            inputs_source, labels_source = iter_source.next()
            inputs_target, labels_target = iter_target.next()

            self.train_batch(inputs_source, labels_source, inputs_target, labels_target, optimizer, loss_config)

        ## TODO:
        return best_model, best_acc

    def train_batch(inputs_source, labels_source, inputs_target, labels_target, optimizer, loss_config):
        ## train one iter
        transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(labels_source)
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features, outputs = self.net(inputs)

        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = loss.MMDLoss()
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)
        ## TODO: loss_config
        transfer_loss = transfer_criterion(features.narrow(0, 0, features.size(0)/2), features.narrow(0, features.size(0)/2, features.size(0)/2), **loss_config["params"])
                    
        total_loss = loss_config["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

    ##  TODO:
    def evaluate():
        pass

