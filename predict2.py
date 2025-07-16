import os
import time
import datetime
from PIL import Image
import numpy as np
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import BaseDataset
from torchvision import transforms as T
from unet import UNet
from unet_2plus import UNet_2Plus
from attention_unet import AttentionUNet
from transUNet import TransUNet as transUNet
from swin_unet import SwinUnet as swin_unet
from da_transUNet import DATransUNet
from msf_transUNet import MSF_transUNet
from sc_transunet import CONFIGS as CONFIGS_ViT_seg
from saunet import SAUNet
from cell_unet import CellUNet
from c_resunet import CResUNet
from sc_transunet import SCTransUNet


def create_config(model_name):
    # UNet, SAUNet, CellUNet, CResUNet, SCTransUNet
    if model_name == "SCTransUNet":
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
    elif model_name == "SAUNet":
        model = SAUNet(in_channels=3, base_dim=64, num_classes=num_classes)
    elif model_name == "CellUNet":
        model = CellUNet(in_channels=3, base_dim=64, num_classes=num_classes)
    elif model_name == "CResUNet":
        model = CResUNet(in_channels=3, base_dim=16, num_classes=num_classes, mode='maxpool')
    elif model_name == "SCTransUNet":
        config_vit.n_classes = num_classes
        model = SCTransUNet(config_vit, 224, num_classes=num_classes)
    else:
        raise ValueError("model name is error!")
    return model


def get_imageList(image_root):
    image_list = []
    for name in os.listdir(image_root):
        cell = name.split('_')[0]
        if cell == "MCF7" or cell == "BMB":
            image_list.append(os.path.join(image_root, name))
    return image_list


def predict(args):
    num_classes = args.num_classes + 1  # include background

    if args.dataset == "A549":
        mean = (0.604, 0.604, 0.603)
        std = (0.055, 0.054, 0.058)
    elif args.dataset == "liveCell":
        mean = (0.502, 0.502, 0.502)
        std = (0.039, 0.039, 0.039)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # get devices
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    image_root = os.path.join(args.data_path, "test/images")
    # load image
    image_list = get_imageList(image_root)
    data_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=mean, std=std)])

    model_list = ['UNet', 'SAUNet', 'CellUNet', 'CResUNet', 'SCTransUNet']

    for model_name in model_list:
        last_weight = "./save_weights/" + args.dataset + '_' + model_name + "_enhance.pth"
        best_weight = "./save_weights/" + args.dataset + "_best" + model_name + "_enhance.pth"
        save_root_path = "./predict/enhance_" + args.dataset
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        save_root_path = os.path.join(save_root_path, model_name)
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        model = create_model(num_classes, model_name)
        model.load_state_dict(torch.load(best_weight if os.path.exists(best_weight) else last_weight,
                                         map_location='cpu')['model'])
        model.to(device)
        for image_name in image_list:
            name = image_name.split('/')[-1]
            original_img = Image.open(image_name).convert('RGB')
            # from pil image to tensor and normalize
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            model.eval()  # 进入验证模式
            with torch.no_grad():
                output = model(img.to(device))
                prediction = output.argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255
                mask = Image.fromarray(prediction)
                save_path = os.path.join(save_root_path, name.split('.')[0] + "_mask_predict.png")
                mask.save(save_path)
                print(save_path + " is saved successful!")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("--dataset", default="liveCell", type=str,
                        help="the current dataset's name")
    parser.add_argument("--data-path", default="./DRIVE224_2", help="DRIVE root")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)
