from torch.autograd import Variable
import torch.onnx
import torchvision


# model_path = '/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth'
# onnx_model_path = "restnet101_0.96.onnx.pb"
# state_dict = torch.utils.model_zoo.load_url(model_path, model_dir="/Users/yangyucheng/Desktop/BirdRecognition")
# dummy_input = Variable(torch.randn(1, 3, 224, 224))
# torch.onnx.export(model, dummy_input, "resnet.proto", verbose=True)

input_shape = (3, 224, 224)
model_onnx_path = "torch_model.onnx"
# model = torch.load('/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth',map_location=lambda storage, location: storage)
# model = torch.load('/Users/yangyucheng/Desktop/BirdRecognition/restnet101_0.96.pth',map_location='cpu')
# net = torch.load('../restnet101_0.96.pth')
model = torchvision.models.densenet161(pretrained=False)
model.classifier = torch.nn.Linear(2208,15)
model.load_state_dict(torch.load('../densnet_0.94_dict.pth'))
# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch.onnx.export(model,dummy_input,model_onnx_path,verbose=False)
print("Export of torch_model.onnx complete!")