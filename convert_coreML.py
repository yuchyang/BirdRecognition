import torch
net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 100}
torch.save(state,'restnet101_0.96.pth')
