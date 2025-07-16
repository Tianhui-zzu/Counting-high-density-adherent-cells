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
from msf_transUNet import CONFIGS as CONFIGS_ViT_seg


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
    return model


def get_imageList(image_root):
    image_list = []
    for name in os.listdir(image_root):
        cell = name.split('_')[0]
        if cell == "MCF7" or cell == "A549":
            image_list.append(os.path.join(image_root, name))
    return image_list


def predict(args):
    num_classes = args.num_classes + 1  # include background

    if args.dataset == "A549":
        mean = (0.604, 0.604, 0.603)
        std = (0.055, 0.054, 0.058)
    else:
        mean = (0.502, 0.502, 0.502)
        std = (0.039, 0.039, 0.039)

    # get devices
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    image_root = os.path.join(args.data_path, "test/images")
    # load image
    image_list = get_imageList(image_root)
    data_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=mean, std=std)])

    model_list = ['UNet', 'UNetPP', 'AttentionUNet', 'TransUNet', 'SwinUNet',
                  'MSFTransUNet', 'DATransUNet']
    for i in range(1, 3):
        for model_name in model_list:
            if i == 1:
                last_weight = "./save_weights/" + args.dataset + '_' + model_name + ".pth"
                best_weight = "./save_weights/" + args.dataset + '_best' + model_name + ".pth"
                save_root_path = "./predict/origin2"
            else:
                last_weight = "./save_weights/" + args.dataset + '_' + model_name + "_enhance.pth"
                best_weight = "./save_weights/" + args.dataset + "_best" + model_name + "_enhance.pth"
                save_root_path = "./predict/enhance2"
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
    parser.add_argument("--dataset", default="A549", type=str,
                        help="the current dataset's name")
    parser.add_argument("--data-path", default="./DRIVE224_3", help="DRIVE root")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)
