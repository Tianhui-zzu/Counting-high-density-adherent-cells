import os
import time
import datetime

import numpy as np
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import BaseDataset
import transforms as T
from baseline import Baseline
from baseline import CONFIGS as CONFIGS_ViT_seg


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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
    return model


def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # classes + background
    class_nums = args.num_classes + 1
    if args.dataset == "A549":
        mean = (0.604, 0.604, 0.603)
        std = (0.055, 0.054, 0.058)
    else:
        mean = (0.502, 0.502, 0.502)
        std = (0.039, 0.039, 0.039)

    test_dataset = BaseDataset(args.data_path, split='test', transform=get_transform(mean=mean, std=std),
                               enhance=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=0,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    model_list = ['Baseline1', 'Baseline2', 'Baseline3', 'Baseline4']

    for model_name in model_list:
        last_weight_path = './save_weights/' + args.dataset + '_' + model_name + '.pth'
        best_weight_path = './save_weights/' + args.dataset + '_best' + model_name + '.pth'

        model = create_model(num_classes=class_nums, model_name=model_name)
        model.load_state_dict(
            torch.load(best_weight_path if os.path.exists(best_weight_path) else last_weight_path,
                       map_location='cpu')[
                'model'])
        model.to(device)

        confmat, dice, _, accuracy, precise, recall, f1 = evaluate(model, test_loader, device, class_nums)

        mIoU = confmat.return_mIoU()
        accuracy = accuracy * 100
        precise = precise * 100
        recall = recall * 100
        f1 = f1 * 100
        dice = dice * 100
        mIoU = mIoU * 100
        print(args.dataset + ' dataset: ' + model_name)
        print(f'mIoU: {mIoU:.2f}')
        print(f'dice coefficient: {dice:.2f}')
        print(f'counting accuracy: {accuracy:.2f}, '                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
              f'precise: {precise:.2f}, '
              f'recall: {recall:.2f}, f1: {f1:.2f}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("--dataset", default="A549", type=str,
                        help="the current dataset's name")
    parser.add_argument("--data-path", default="./DRIVE224_3", help="DRIVE root")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
