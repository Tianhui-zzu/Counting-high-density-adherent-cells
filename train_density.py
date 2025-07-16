import os
import time
import datetime

import numpy as np
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import BaseDataset
import transforms as T
from unet import UNet
from unet_2plus import UNet_2Plus
from attention_unet import AttentionUNet
from transUNet import TransUNet as transUNet
from swin_unet import SwinUnet as swin_unet
from da_transUNet import DATransUNet
from msf_transUNet import MSF_transUNet
from msf_transUNet import CONFIGS as CONFIGS_ViT_seg


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.8 * base_size)
        max_size = int(1.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 280
    crop_size = 224

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_config(model_name):
    if model_name == "SwinUNet":
        vit_config = CONFIGS_ViT_seg['SWIN_T_TINY']
    elif model_name == "MSFTransUNet":
        vit_config = CONFIGS_ViT_seg['GC_Vit_base']
    elif model_name == "TransUNet" or model_name == "DATransUNet":
        vit_config = CONFIGS_ViT_seg['R50-ViT-B_16']
        vit_config.n_skip = 3
        vit_config.patches.grid = (14, 14)
    else:
        vit_config = None
    return vit_config


def create_model(num_classes, model_name):
    config_vit = create_config(model_name)
    if model_name == "UNet":
        model = UNet(in_channels=3, base_dim=64, num_classes=num_classes, bilinear=False, mode='conv')
    elif model_name == "UNetPP":
        model = UNet_2Plus(in_channels=3, n_classes=2)
    elif model_name == "AttentionUNet":
        model = AttentionUNet(3, num_classes)
    elif model_name == "TransUNet":
        config_vit.n_classes = num_classes
        model = transUNet(config_vit, 224, num_classes)
    elif model_name == "SwinUNet":
        model = swin_unet(config_vit, img_size=224, num_classes=num_classes)
    elif model_name == "DATransUNet":
        model = DATransUNet(config_vit, 224, num_classes)
    elif model_name == "MSFTransUNet":
        model = MSF_transUNet(config_vit, num_classes)
    else:
        raise ValueError("model name is error!")
    if config_vit is not None:
        if model_name == "TransUNet" or model_name == "DATransUNet":
            model.load_from(weights=np.load(config_vit.pretrained_path))
        else:
            model.load_from(config_vit)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # classes + background
    class_nums = args.num_classes + 1
    batch_size = args.batch_size
    if args.dataset == "A549":
        mean = (0.604, 0.604, 0.603)
        std = (0.055, 0.054, 0.058)
    else:
        mean = (0.502, 0.502, 0.502)
        std = (0.039, 0.039, 0.039)
    if args.enhance:

        last_weight = "./save_weights/" + args.dataset + '_' + args.model + '_density' + args.density + '_enhance.pth'
        best_weight = "./save_weights/" + args.dataset + "_best" + args.model + '_density' + args.density + '_enhance.pth'
        save_model_name = args.model + '_density' + args.density + "_enhance"
    else:
        last_weight = "./save_weights/" + args.dataset + '_' + args.model + '_density' + args.density + '.pth'
        best_weight = "./save_weights/" + args.dataset + "_best" + args.model + '_density' + args.density + '.pth'
        save_model_name = args.model + '_density' + args.density

    results_file = save_model_name + "_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data_path = os.path.join(args.data_path, args.density)
    train_dataset = BaseDataset(data_path, split='training',
                                transform=get_transform(train=True, mean=mean, std=std), enhance=args.enhance)
    val_dataset = BaseDataset(data_path, split='val', transform=get_transform(train=False, mean=mean, std=std),
                              enhance=args.enhance)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=class_nums, model_name=args.model)
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # create lr update
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
        else:
            best_dice = 0.
        if args.amp:
            scaler.load_state_dict(checkpoint['scaler'])
    else:
        best_dice = 0.
    start_time = time.time()
    save = False
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, class_nums,
                                        lr_scheduler=lr_scheduler,
                                        print_freq=args.print_freq, scaler=scaler)

        confmat, dice, dsc, accuracy, precise, recall, f1 = evaluate(model, val_loader, device, class_nums)
        val_info = str(confmat)
        print(val_info)
        print(f'dice coefficient: {dice:.4f}')
        print(f'counting accuracy: {accuracy:.4f}, precise: {precise:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
        with open(results_file, 'a') as f:
            # save every train epoch loss, lr
            train_info = f'[epoch: {epoch}]\n' \
                         f'[train_loss: {mean_loss:.4f}]\n' \
                         f'[lr: {lr:.6f}]\n' \
                         f'[dice coefficient: {dice:.4f}]\n' \
                         f'[accuracy: {accuracy:.4f}]\n' \
                         f'[precise: {precise:.4f}]\n' \
                         f'[recall: {recall:.4f}]\n' \
                         f'[f1: {f1:.4f}]\n'

            f.write(train_info + val_info + '\n\n')
        if args.save_best:
            if best_dice < dice:
                save = True
                best_dice = dice
            else:
                save = False
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args,
                     "best_dice": best_dice
                     }
        if args.amp:
            save_file['scaler'] = scaler.state_dict()
        # save newest weight
        if epoch % 5 == 0:
            torch.save(save_file, last_weight)
        if save:
            torch.save(save_file, best_weight)
    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print('total training time is {} seconds'.format(total_time))

# ./save_weights/A549_MSFTransUNet_density40_enhance.pth

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=24, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to training")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='./save_weights/A549_MSFTransUNet_density40_enhance.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--model", default="MSFTransUNet", type=str,
                        help="currently use model name (UNet, UNetPP, AttentionUNet, TransUNet, SwinUNet, "
                             "MSFTransUNet, DATransUNet)")
    parser.add_argument("--dataset", default="A549", type=str,
                        help="the current dataset's name, liveCell or A549")
    parser.add_argument("--data-path", default="./density", help="DRIVE root")
    parser.add_argument("--density", default="40", help="the density of cell")
    parser.add_argument("--enhance", default=True, type=bool,
                        help="if use the data enhancement")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
