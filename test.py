import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset

from dataset.ISBI_datasert import UnetTestDataset, transform_image
from model import *
import argparse


ori_img_path = "./data/membrane/test"
test_img_list = sorted(os.listdir(ori_img_path))
# print(ori_list)




parser = argparse.ArgumentParser(description='PyTorch Unet ISBI Challenge')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--save_folder', default='./results/',
                        help='Directory for saving checkpoint models')
args = parser.parse_args()
if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


test_dataset = UnetTestDataset(img_root= ori_img_path,img_list=test_img_list, transform=transform_image)


kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = UNet(1, 1)
model.load_state_dict(torch.load('./checkpoints/my_unet.pth'))
model = model.to(device)
pic = []
m = nn.Sigmoid()

def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, name) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            output = m(output)
            # print(output.shape)
            pic.append((output, name))
    return pic

def main():
    test(args, model, device, test_loader)
    i=0
    for img_tensor in pic:
        img = img_tensor[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * 255
        # print(img.shape)
        # img = Image.fromarray(img)
        cv2.imwrite("./results/%s.jpg" % (img_tensor[1]), img)
        i = i+1

if __name__ == '__main__':
    main()