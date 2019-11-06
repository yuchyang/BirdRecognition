from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
import cv2
import time

class CUB_200(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CUB_200, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classes_file = os.path.join(root, "classes_txt")
        self.image_class_labels_file = os.path.join(root, "image_class_labels.txt")
        self.images_file = os.path.join(root, "images.txt")
        self.train_test_split_file = os.path.join(root, "train_test_split.txt")
        self.bounding_boxes_file = os.path.join(root, "bounding_boxes.txt")


        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}

        self._image_id_loc = {}

        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception(' label Error! ')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

        for line in open(self.bounding_boxes_file):
            image_id, x, y, width, height = line.strip('\n').split()
            self._image_id_loc[image_id] = x, y, width, height

    def _get_PIL_image(self,image_path):
        image = Image.open(image_path).convert('RGB')
        return image
    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label, image_id))
            else:
                self._test_path_label.append((image_name, label, image_id))

    def __getitem__(self, index):
        if self.train:
            image_name, label, image_id = self._train_path_label[index]
        else:
            image_name, label, image_id = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        # img = Image.open(image_path)
        # if img.mode == 'L':
        #     img = img.convert('RGB')
        label = int(label) - 1
        x, y, width, height = self._image_id_loc[image_id]
        boxes = []
        # x1, y1, x2, y2 = x, y, x + width, y+height
        x1 = float(x)
        y1 = float(y)
        x2 = x1 + float(width)
        y2 = y1 + float(height)
        boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes, dtype=np.float32)
        # if self.target_transform:
        labels = []
        labels.append(label)
        labels = np.array(labels, dtype=np.int64)
        # print('boxes shape')
        # print(boxes.shape)
        # print(boxes)
        # print(img.size)

        # print('labels shape1',labels.shape)
        # print('boxes shape1',boxes.shape)
        if self.transform:
            if self.target_transform:
                img = self._read_image(image_path)
                img, boxes, labels = self.transform(img, boxes, labels)
            else:
                labels = labels[0]
                img = self._get_PIL_image(image_path)
                img = self.transform(img)
        # print('labels shape2', labels.shape)
        # print('boxes shape2', boxes.shape)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        # print('labels shape3', labels.shape)
        # print('boxes shape3', boxes.shape)
        return img, labels, boxes

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    cub200_root = "D:\BirdRecognition\CUB_200_2011"

    cub = CUB_200(cub200_root, train=True, transform=transform)
    # for img, label, box in cub:
    #     #     print(img.size(), label, box)
    #     #     if img.size(0) != 3:
    #     #         raise ValueError("????  3333 ")
