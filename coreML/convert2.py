from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from vision.nn.mobilenet_v2 import MobileNetV2, InvertedResidual

from vision.ssd.ssd import SSD, GraphPath
from vision.ssd.predictor import Predictor
from vision.ssd.config import mobilenetv1_ssd_config as config
from SSD.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite


# model_path = '/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth'
# onnx_model_path = "restnet101_0.96.onnx.pb"
# state_dict = torch.utils.model_zoo.load_url(model_path, model_dir="/Users/yangyucheng/Desktop/BirdRecognition")
# dummy_input = Variable(torch.randn(1, 3, 224, 224))
# torch.onnx.export(model, dummy_input, "resnet.proto", verbose=True)

input_shape = (3, 300, 300)
model_onnx_path = "mobile_ssd.onnx"
# model = torch.load('/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth',map_location=lambda storage, location: storage)
# model = torch.load('/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth',map_location='cpu')
# net = torch.load('../restnet101_0.96.pth')
net = create_mobilenetv2_ssd_lite(200, is_test=True, onnx_compatible=True)
net.load('D://BirdRecognition//mobilenetv2-Epoch-106-Loss-6.8959480877761.pth')
net.cuda()

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape)).cuda()
output = torch.onnx.export(net,dummy_input,model_onnx_path,verbose=False)
print("Export of torch_model.onnx complete!")