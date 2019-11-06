import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from layer import grl
import model.backbone as backbone
from torch.autograd import Variable
from coreML.DANN import DANN


input_shape = (3, 224, 224)
model_onnx_path = "DANN200.onnx"
c_net = torch.load('D:\BirdRecognition\domain_adaptation\DANN_30spc_accuracy0.9564250778123611_c_net')
model = DANN(base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=200, hidden_dim=1024, trade_off=1.0, use_gpu=False)
# model.load_model('D:\BirdRecognition\domain_adaptation\DANN_30spc_accuracy0.9564250778123611_c_net', 'D:\BirdRecognition\domain_adaptation\DANN_30s[cE_accuracy0.9564250778123611_d_net')
model.load_model('D:\BirdRecognition\domain_adaptation\DANN_IMAGE200_accuracy0.7355885398688298_c_net', 'D:\BirdRecognition\domain_adaptation\DANN_IMAGE200_accuracy0.7355885398688298_d_net')
model.set_train(False)
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch.onnx.export(model,dummy_input,model_onnx_path,verbose=False)
print("Export of onnx complete!")