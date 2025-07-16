import os
import time
import datetime
import random
import numpy as np
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import BaseDataset
import transforms as T
from baseline import Baseline
from baseline import CONFIGS as CONFIGS_ViT_seg


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


def create_config():
    vit_config = CONFIGS_ViT_seg['GC_Vit_base']
    return vit_config


def create_model(num_classes, model_name):
    config_vit = create_config()
    if model_name == "Baseline1":
        model = Baseline(config_vit, num_classes, False, False, False)
    elif model_name == "Baseline2":
        model = Baseline(config_vit, num_classes, True, False, False)
    elif model_name == "Baseline3":
        model = Baseline(config_vit, num_classes, True, True, False)
    elif model_name == "Baseline4":
        model = Baseline(config_vit, num_classes, True, True, True)
    else:
        raise ValueError("model name is error!")
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
        last_weight = "./save_weights/" + args.dataset + '_' + args.model + '_enhance.pth'
        best_weight = "./save_weights/" + args.dataset + "_best" + args.model + '_enhance.pth'
        save_model_name = args.model + "_enhance"
    else:
        last_weight = "./save_weights/" + args.dataset + '_' + args.model + '.pth'
        best_weight = "./save_weights/" + args.dataset + "_best" + args.model + '.pth'
        save_model_name = args.model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    results_file = save_model_name + "_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = BaseDataset(args.data_path, split='training',
                                transform=get_transform(train=True, mean=mean, std=std), enhance=args.enhance)
    val_dataset = BaseDataset(args.data_path, split='val', transform=get_transform(train=False, mean=mean, std=std),
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
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--model", default="Baseline4", type=str,
                        help="Baseline1(+GC-ViT),Baseline2(+GC-ViT+MK_Conv),Baseline3(+GC-ViT+MK_Conv+BottleNeck),"
                             "Baseline4(+GC-ViT+MK_Conv+BottleNeck+MSFF)")
    parser.add_argument("--dataset", default="A549", type=str,
                        help="the current dataset's name")
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument("--data-path", default="./DRIVE224_3", help="DRIVE root")
    parser.add_argument("--enhance", default=False, type=bool,
                        help="if use the data enhancement")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
