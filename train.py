import os
import torch.optim as optim
from torch.utils import data
from dataset.ISBI_datasert import UnetDataset, transform_image, transform_label

from model import *
import argparse

dataset_img_root = "./data/membrane/train/image"
ori_list = sorted(os.listdir(dataset_img_root))
# print(ori_list)
train_img_list, val_img_list = ori_list[:24], ori_list[24:]

dataset_label_root = "./data/membrane/train/label"
label_list = sorted(os.listdir(dataset_label_root))
# print(label_list)
train_label_list, val_label_list = label_list[:24], label_list[24:]

parser = argparse.ArgumentParser(description='PyTorch Unet ISBI Challenge')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
parser.add_argument('--val_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
parser.add_argument('--log_interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--save_model_name', default='my_unet.pth',
                        help='name of saved model')
parser.add_argument('--save_folder', default='./checkpoints/',
                        help='Directory for saving checkpoint models')
parser.add_argument('--test_value',default=0)

args = parser.parse_args()

# print(f'batch_size={args.batch_size}')
criterion = nn.BCEWithLogitsLoss() #计算真实的标签和我预测的标签之间的差距，差距越大loss越大，我们要追求一个最小的loss
sig = nn.Sigmoid()
best_dice = 0.0


def main():
    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = UnetDataset(img_root=dataset_img_root, label_root=dataset_label_root, img_list=train_img_list,
                                label_list=train_label_list,
                                transform=transform_image, target_transform=transform_label)
    val_dataset = UnetDataset(img_root=dataset_img_root, label_root=dataset_label_root, img_list=val_img_list,
                              label_list=val_label_list,
                              transform=transform_image, target_transform=transform_label)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size, shuffle=False, num_workers=8,
                                             pin_memory=False)


    model = UNet(1, 1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(args.test_value)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        val(args, model, device, val_loader)

# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    dice = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print("data shape:{}".format(data.dtype))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) # This loss is per image
        sig_output = sig(output)
        dice += dice_coeff(sig_output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    dice_acc = 100. * dice / len(train_loader)
    print('Train Dice coefficient: {:.2f}%'.format(dice_acc))

def val(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    dice = 0
    global best_dice
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            sig_output = sig(output)
            dice += dice_coeff(sig_output, target)


    val_loss /= len(val_loader.dataset)
    dice_acc = 100. * dice / len(val_loader)

    print('\nVal set: Batch average loss: {:.4f}, Dice Coefficient: {:.2f}%\n'.format(val_loss, dice_acc))

    if dice_acc > best_dice:
        torch.save(model.state_dict(), args.save_folder + args.save_model_name)
        best_dice = dice_acc
        print("======Saving model======")

if __name__ == '__main__':
    print('1')
    main()