import torch
import torch.nn as nn
import layer
from loss import loss
import model.backbone as backbone

class JAN_Net(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31):
        super(JAN_Net, self).__init__()

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

class JAN(object):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, class_num=31, 
kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68], trade_off=1.0, use_gpu=True):
        self.net = JAN_Net(base_net, use_bottleneck, bottleneck_dim, class_num)
        self.kernel_muls = kernel_muls
        self.kernel_nums = kernel_nums
        self.fix_sigma_list = fix_sigma_list
        self.trade_off = trade_off
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.net = self.net.cuda()
        self.is_train = False

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = loss.jmmd_loss
        features, outputs, softmax_outputs = self.net(inputs)
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)
        transfer_loss = transfer_criterion([features.narrow(0, 0, features.size(0)/2), 
                            softmax_outputs.narrow(0, 0, softmax_outputs.size(0)/2)], 
                    [features.narrow(0, features.size(0)/2, features.size(0)/2), 
            softmax_outputs.narrow(0, softmax_outputs.size(0)/2, softmax_outputs.size(0)/2)], 
 kernel_muls=self.kernel_muls, kernel_nums=self.kernel_nums, fix_sigma_list=self.fix_sigma_list)

        total_loss = self.trade_off * transfer_loss + classifier_loss
        return total_loss

    def predict(self, inputs):
        _, _, softmax_outputs = self.net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.net.parameter_list

    def save_model(self, model_path):
        torch.save(self.net.state_dict(), model_path)

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))

    def set_train(self, mode):
        self.net.train(mode) 
        self.is_train = mode

