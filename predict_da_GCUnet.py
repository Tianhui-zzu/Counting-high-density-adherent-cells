import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from da_gcUnet2 import DAGCUnet as ViT_seg
from model import SA_Unet
from da_transUNet import CONFIGS as CONFIGS_ViT_seg
from da_gcUnet2 import CONFIGS as CONFIGS_ViT_seg


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_imageList(image_root):
    image_list = []
    # 1389
    for name in os.listdir(image_root):
        cell = name.split('_')[0]
        if cell == 'BV2':
            image_list.append(os.path.join(image_root, name))
    return image_list


def create_model(config_vit, num_classes):
    model = ViT_seg(config_vit, num_classes)
    return model


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/DA_BestGCUnet.pth"
    img_path = "./DRIVE/test/images/img_4.png"
    # roi_mask_path = "./DRIVE/test/masks/img_4_mask.png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.502, 0.502, 0.502)
    std = (0.039, 0.039, 0.039)
    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    num_classes = classes + 1
    config_vit = CONFIGS_ViT_seg['GC_ViT_small']
    model = create_model(config_vit=config_vit, num_classes=num_classes)
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)
    image_root = './DRIVE224_1/test/images'
    # load image
    image_list = get_imageList(image_root)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    if not os.path.exists('./predict'):
        os.mkdir('./predict')
    elif not os.path.exists('./predict/predict_da_GCUnet'):
        os.mkdir('./predict/predict_da_GCUnet')
    for image_name in image_list:
        # 1.png
        name = image_name.split('/')[-1].split('.')[0]
        # print(name)
        original_img = Image.open(image_name).convert('RGB')

        # from pil image to tensor and normalize
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            # prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save(os.path.join('./predict/predict_da_GCUnet', name + '_mask_predict.png'))


if __name__ == '__main__':
    main()
