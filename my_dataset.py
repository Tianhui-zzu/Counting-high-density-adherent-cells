import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    base_dir: the location of images and masks
    split: training or val or test
    transform: transform for images and masks, such as ratio, Flip horizontally, eta
    """

    def __init__(self, base_dir, split, transform=None, enhance=False):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        if self.split == 'training':
            self.images_dir = os.path.join(base_dir, 'training', 'images')
            self.masks_dir = os.path.join(base_dir, 'training', 'masks')
        elif self.split == 'val':
            self.images_dir = os.path.join(base_dir, 'val', 'images')
            self.masks_dir = os.path.join(base_dir, 'val', 'masks')
        elif self.split == 'test':
            self.images_dir = os.path.join(base_dir, 'test', 'images')
            self.masks_dir = os.path.join(base_dir, 'test', 'masks')
        else:
            raise Exception("parameter 'split' is only training or val or test!")
        if enhance:
            self.masks_dir = os.path.join(base_dir, self.split, "enhanceMasks")
        if os.path.exists(self.images_dir) is False or os.path.exists(self.masks_dir) is False:
            raise FileNotFoundError("image_dir os masks_dir is not found!")
        images_name = os.listdir(self.images_dir)
        self.images_list = [os.path.join(self.images_dir, i) for i in images_name if
                            i.endswith('.png') and (
                                    i.split('_')[0] == 'MCF7' or i.split('_')[0] == 'A549' or i.split('_')[
                                0] == 'BMB')]
        if enhance:
            self.masks_list = [os.path.join(self.masks_dir, os.path.splitext(i)[0] + '_mask_enhance.png') for i in
                               images_name
                               if i.endswith('.png') and (
                                       i.split('_')[0] == 'MCF7' or i.split('_')[0] == 'A549' or i.split('_')[
                                   0] == 'BMB')]
        else:
            self.masks_list = [os.path.join(self.masks_dir, os.path.splitext(i)[0] + '_mask.png') for i in images_name
                               if i.endswith('.png') and (
                                       i.split('_')[0] == 'MCF7' or i.split('_')[0] == 'A549' or i.split('_')[
                                   0] == 'BMB')]

        for i in self.masks_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} is not exists!")

    def __getitem__(self, item):
        img = Image.open(self.images_list[item]).convert('RGB')
        mask = Image.open(self.masks_list[item]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
