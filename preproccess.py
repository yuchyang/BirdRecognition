import torch
import torchvision
from torchvision import utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

img_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition',
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
                                            )

print(len(img_data))
data_loader = torch.utils.data.DataLoader(img_data, batch_size=20, shuffle=True)
print(len(data_loader))


def show_batch(imgs):
    grid = utils.make_grid(imgs, nrow=5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if i < 4:
            print(i, batch_x.size(), batch_y.size())
s