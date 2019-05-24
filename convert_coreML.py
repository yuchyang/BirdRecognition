import torch
net = torch.load('densnet_0.9433333333333334.pkl')
print(net)
# torch.save(net.state_dict(),'densnet_0.94_dict.pth')
