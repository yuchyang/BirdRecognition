import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0)
    print(img.shape)

    return img

def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = torchvision.transforms.ToPILImage()(img)
    img.show()
    # plt.figure()
    # plt.imshow(img)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)

class Padding(object):
    def __call__(self, img):
        w, h = img.size
        if w>h:
            if (w-h)%2 != 0:
                padding = (0, (w - h) // 2 + 1, 0, (w - h) // 2)
            else:
                padding = (0, (w - h) // 2, 0, (w - h) // 2)

        elif w<h:
            if (h-w)%2 != 0:
                padding = ((h - w) // 2 + 1, 0, (h - w) // 2 , 0)
            else:
                padding = ((h - w) // 2, 0, (h - w) // 2, 0)
        else:
            padding = (0,0,0,0)
        img = torchvision.transforms.functional.pad(img,padding,padding_mode='reflect')
        return img