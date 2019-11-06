import torch
from SSD.cub import CUB_200
from torch.utils.data import DataLoader
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config
import torch
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import cv2
from SSD.cub import CUB_200
from utils import utils
from vision.ssd.ssd import MatchPrior
import logging
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

DEVICE = 'cuda'
batch_size = 32

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, labels, boxes = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # print(boxes)
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            # logging.info(
            #     f"Epoch: {epoch}, Step: {i}, " +
            #     f"Average Loss: {avg_loss:.4f}, " +
            #     f"Average Regression Loss {avg_reg_loss:.4f}, " +
            #     f"Average Classification Loss: {avg_clf_loss:.4f}"
            # )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):

    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, labels, boxes = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    #         raise ValueError("????  3333 ")
    config = mobilenetv1_ssd_config
    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=1,onnx_compatible=True)
    # create_net = create_mobilenetv1_ssd
    cub200_root = "D:\BirdRecognition\CUB_200_2011"
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomCrop(300),
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)

    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    cub =CUB_200(cub200_root, train=True, transform=train_transform, target_transform=target_transform)
    # for img, label, box in cub:
    #     print(img.size(), label, box)
    #     if img.size(0) != 3:
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    train_loader = torch.utils.data.DataLoader(cub, batch_size=batch_size, shuffle=True)
    test_dataset = CUB_200(cub200_root, train=False, transform=test_transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    net = create_net(200)
    net.to(DEVICE)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    params = [
        {'params': net.base_net.parameters(), 'lr': 0.001},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': 0.001},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9,
                                weight_decay=0.005)
    # logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
    #              + f"Extra Layers learning rate: {extra_layers_lr}.")
    config = mobilenetv1_ssd_config

    milestones = [int(v.strip()) for v in "80,100".split(",")]
    scheduler = MultiStepLR(optimizer, milestones=milestones,
                            gamma=0.1, last_epoch=-1)
    for epoch in range(0, 500):
        scheduler.step()
        # train(train_loader, net, criterion, optimizer,
        #       device=DEVICE, debug_steps=100, epoch=epoch)

        if epoch % 5 == 1 or epoch == 5 - 1:
            val_loss, val_regression_loss, val_classification_loss = test(test_loader, net, criterion, DEVICE)
            model_path = os.path.join('D://BirdRecognition//', f"mobilenetv2-Epoch-{epoch}-Loss-{val_loss}.pth")
            # net.save(model_path)
            logging.info(f"Saved model ")
