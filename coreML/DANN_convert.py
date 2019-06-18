import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import model.backbone as backbone
from torch.autograd import Variable
from coreML.DANN import DANN


input_shape = (3, 224, 224)
model_onnx_path = "DANN.onnx"
c_net = torch.load('D:/model/DANN_accuracy0.8743386243386243_c_net_0.9853333333333333.pkl')
model = DANN(base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=15, hidden_dim=1024, trade_off=1.0, use_gpu=False)
model.load_model('D:/model/DANN_accuracy0.8743386243386243_c_net_0.9853333333333333.pkl', 'D:/model/DANN_accuracy0.8743386243386243_d_net')
model.set_train(False)
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch.onnx.export(model,dummy_input,model_onnx_path,verbose=False)
print("Export of onnx complete!")