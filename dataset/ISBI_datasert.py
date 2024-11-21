import os

import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
from PIL import Image

class toBinary(object):
    def __call__(self, label):
        label = np.array(label)
        # print(image)
        label = label * (label > 127)
        label = Image.fromarray(label)
        return label

transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
])

transform_label = transforms.Compose([
    transforms.Grayscale(),
    toBinary(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4938, 0.4933, 0.4880), (0.1707, 0.1704, 0.1672)),
])

class UnetDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root, label_root,
                 img_list=None, label_list=None,
                 transform=None, target_transform=None):
        assert img_root is not None and label_root is not None, 'Must specify img_root and label_root!'
        self.img_root = img_root
        self.label_root = label_root
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = Image.open(os.path.join(self.label_root, self.label_list[index]))
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label
        else:
            return image


    def __len__(self):
        return len(self.img_list)

class UnetTestDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root,
                 img_list=None, transform=None):
        assert img_root is not None, 'Must specify img_root and label_root!'

        self.img_root = img_root
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        image_name = self.img_list[index][:-4]
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name

    def __len__(self):
        return len(self.img_list)