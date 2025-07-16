import os
import torch
from skimage import measure
import cv2
import numpy as np
import math
import shutil
from PIL import Image
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
"""
    function: segmentation image to 224×224
    input: original image
    return: save 224×224 images
"""


def split_image224(root_path, output_dir, flag, crop_size=[224, 224]):
    images_path = os.path.join(root_path, flag, 'images')
    masks_path = os.path.join(root_path, flag, 'masks')
    output_images_path = os.path.join(output_dir, flag, 'images')
    output_masks_path = os.path.join(output_dir, flag, 'masks')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, flag)):
        os.mkdir(os.path.join(output_dir, flag))
        os.mkdir(output_images_path)
        os.mkdir(output_masks_path)

    # if not os.path.exists(output_images_path):
    #     os.mkdir(output_images_path)
    # if not os.path.exists(output_masks_path):
    #     os.mkdir(output_masks_path)
    image_list = os.listdir(images_path)
    for img_name in image_list:
        image = cv2.imread(os.path.join(images_path, img_name))
        name = os.path.splitext(img_name)[0]
        mask_name = name + '_mask.png'
        mask = cv2.imread(os.path.join(masks_path, mask_name))
        height, width = image.shape[0:2]
        step_size = 224
        num = 1
        for i in range(0, height - crop_size[1] + 1, step_size):
            for j in range(0, width - crop_size[0] + 1, step_size):
                new_image = image[i:i + crop_size[1], j:j + crop_size[0], :]
                new_mask = mask[i:i + crop_size[1], j:j + crop_size[0], :]
                new_img_name = name + '_' + str(num) + '.png'
                new_msk_name = name + '_' + str(num) + '_mask.png'
                cv2.imwrite(os.path.join(output_images_path, new_img_name), new_image)
                cv2.imwrite(os.path.join(output_masks_path, new_msk_name), new_mask)
                print('{} save finish'.format(new_img_name))
                print('{} save finish'.format(new_msk_name))
                num += 1


def remove_small_points(image, threshold_point):
    original_img = cv2.imread(image)
    img = original_img[:, :, 0] // 255 if len(original_img.shape) == 3 else original_img[:, :] // 255
    img_label, num = measure.label(img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)  # 创建一个原图像大小的全零数组
    for i in range(0, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix[resMatrix > 0] = 1
    resMatrix *= 255
    result = np.zeros(original_img.shape)
    if len(original_img.shape) == 3:
        result[:, :, 0] = resMatrix
        result[:, :, 1] = resMatrix
        result[:, :, 2] = resMatrix
    else:
        result = resMatrix
    return result


def postprocess(image, save_path):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = image[:, :, 0] if len(image.shape) == 3 else image
    image[image == 1] = 255
    kernel = np.ones((3, 3), np.uint8)
    rows, cols = kernel.shape
    new_kernel = []
    for i in range(rows):
        result = [0 if math.sqrt((i - 1) ** 2 + (j - 1) ** 2) > 1 else 1 for j in range(cols)]
        new_kernel.append(result)
    new_kernel = np.array(new_kernel, dtype=np.uint8)
    target = cv2.erode(image, new_kernel, 1)
    cv2.imwrite(save_path, target)


def run_postprocess():
    image_root = '../predict/postprocess/MCF7_173_4_mask_predict_final.png'
    save_root = '../predict/postprocess'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = '../predict/postprocess/MCF7_173_4_mask_predict_final_postprocess.png'
    postprocess(image_root, save_path)


def run_split_image224():
    output_dir = '../DRIVE224_1'
    root_path = '../DRIVE1'
    flags = ['training', 'val', 'test']
    for flag in flags:
        split_image224(root_path=root_path, output_dir=output_dir, flag=flag)


def run_remove_small_points():
    root_path = '../DRIVE224_3'
    flags = ['training', 'val', 'test']
    for flag in flags:
        mask_path = os.path.join(root_path, flag, 'masks')
        mask_list = os.listdir(mask_path)
        for name in mask_list:
            mask_name = os.path.join(mask_path, name)
            cell_name = name.split('_')[0]
            if cell_name == "A549":
                result = remove_small_points(mask_name, 40)
            else:
                result = remove_small_points(mask_name, 10)
            cv2.imwrite(mask_name, result)
            print('{} save finish!'.format(name))


def calculation_cell_counting_accuracy(original_img_path, predict_img_path):
    original_img = cv2.imread(original_img_path)

    # A549: 40  MCF7: 10
    predict_img = remove_small_points(predict_img_path, 10)
    # predict_img = cv2.imread(predict_img_path)
    original_img = original_img[:, :, 0] // 255
    predict_img = predict_img[:, :, 0] // 255 if len(predict_img.shape) == 3 else predict_img[:, :] / 255

    original_img_label, original_num = measure.label(original_img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    predict_img_label, predict_num = measure.label(predict_img, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_img_label)
    predict_props = measure.regionprops(predict_img_label)
    tp = 0
    fp = 0
    original_flags = np.zeros(original_num)
    for i in range(len(predict_props)):
        center_y, center_x = predict_props[i].centroid
        center_x = int(center_x)
        center_y = int(center_y)
        flag = original_img_label[center_y, center_x]
        if flag >= 1 and original_flags[flag - 1] == 0:
            tp += 1
            original_flags[flag - 1] = 1
        else:
            fp += 1
    fn = original_num - tp
    counting_accuracy = ((tp + 0.01) / (tp + fp + fn + 0.01)) * 100
    if counting_accuracy < 70:
        print('{} accuracy: {:.2f}%'.format(predict_img_path, counting_accuracy))
    return counting_accuracy


def run_delete_mask(mask_path):
    mask_list = os.listdir(mask_path)
    for mask_name in mask_list:
        if 'enhance' in mask_name:
            msk_path = os.path.join(mask_path, mask_name)
            os.remove(msk_path)
            print(mask_name + ' is deleted')


# A549/MCF7数据集增强
def data_annotation_enhance(mask_path):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    cell_name = mask_path.split('/')[-1].split('_')[0]
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    areas = []
    for i in range(len(original_props)):
        areas.append(original_props[i].area)
    sort_areas = sorted(areas)
    threshold = sort_areas[original_num // 2]
    for i in range(len(original_props)):
        if original_props[i].area > threshold:
            if cell_name == 'A549':
                radius = 10
            else:
                radius = 4
        else:
            if cell_name == "A549":
                if original_props[i].area <= 50:
                    radius = 5
                else:
                    radius = 7
            else:
                if original_props[i].area <= 15:
                    radius = 2
                else:
                    radius = 3

        cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), radius,
                   (255, 255, 255), -1)
    enhance_mask_name = mask_path.split('/')[-1].split('.')[0] + '_enhance.png'
    enhance_mask_path = '/'.join(mask_path.split('/')[0:-2]) + '/enhanceMasks'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


# BM数据集增强
def data_annotation_enhance_BM(mask_path, num):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    for i in range(len(original_props)):
        cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), 5,
                   (255, 255, 255), -1)
    enhance_mask_name = "BM_" + str(num) + "_enhance.png"
    enhance_mask_path = '/'.join(mask_path.split('/')[0:-1]) + '/enhanceMasks'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


# A549/MCF7增强
def run_data_annotation_enhance():
    mask_root_path = '../DRIVE224_3/'
    flags = ["training", "val", "test"]
    for name in flags:
        mask_path = os.path.join(mask_root_path, name, "masks")
        image_list = os.listdir(mask_path)
        for mask_name in image_list:
            cell = mask_name.split('_')[0]
            if cell == "MCF7" or cell == "A549":
                msk_path = os.path.join(mask_path, mask_name)
                data_annotation_enhance(msk_path)
            else:
                continue


# BM增强
def run_data_annotation_enhance_BM():
    mask_root_path = '../DRIVE2/'
    mask_path = os.path.join(mask_root_path, "masks")
    image_list = os.listdir(mask_path)
    num = 1
    for mask_name in image_list:
        msk_path = os.path.join(mask_path, mask_name)
        data_annotation_enhance_BM(msk_path, num)
        num += 1


def run_calculation_cell_counting_accuracy():
    predict_root = '../predict/predict_da_GCUnet'
    original_root = '../DRIVE224_1/test/masks'
    original_list = os.listdir(original_root)
    total_acc = 0
    num = 0
    for original_name in original_list:
        cell = original_name.split('_')[0]
        if cell == 'BV2':
            num += 1
            predict_name = original_name.split('.')[0] + '_predict.png'
            original_name_path = os.path.join(original_root, original_name)
            predict_name_path = os.path.join(predict_root, predict_name)
            counting_accuracy = calculation_cell_counting_accuracy(original_name_path, predict_name_path)
            total_acc += counting_accuracy
    total_acc = total_acc / num
    print('mean cell counting accuracy: {:.2f}%'.format(total_acc))


def copy_label_files(source_dir, target_dir):
    # 遍历源目录下的所有一级子目录
    for subdir in os.listdir(source_dir):
        sub_path = os.path.join(source_dir, subdir)
        if os.path.isdir(sub_path):
            # 在子目录下查找 label.png 文件
            label_file = os.path.join(sub_path, 'label.png')
            if os.path.isfile(label_file):
                # 将 label.png 复制到目标位置，并以子目录名字命名
                image_name = '_'.join(subdir.split('_')[0:-1])
                target_filename = os.path.join(target_dir, image_name + '.png')
                shutil.copy(label_file, target_filename)
                print(f"复制 {label_file} 到 {target_filename}")


def convertAnn(mask_path):
    origin_mask = cv2.imread(mask_path)
    mask = np.zeros(origin_mask.shape, dtype="uint8")
    for i in range(origin_mask.shape[0]):
        for j in range(origin_mask.shape[1]):
            if origin_mask[i, j, 0] > 0 or origin_mask[i, j, 1] > 0 or origin_mask[i, j, 2] > 0:
                mask[i, j, :] = 255
    new_path = os.path.join('/'.join(mask_path.split('/')[0:-1]), "label_convert.png")
    cv2.imwrite(new_path, mask)


def compute_mean_std(root_path):
    img_channels = 3
    img_dir = root_path
    # roi_dir = "./DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    # assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        # ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        # roi_img = np.array(Image.open(ori_path).convert('L'))
        img = img.reshape(-1, 3)
        # img = img[roi_img == 255]
        # test = img.mean(axis=0)
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    mean = np.round(mean, 3)
    std = np.round(std, 3)
    print(f"mean: {mean}")
    print(f"std: {std}")


def drawResultFigure(image_path, mask_path, origin_path, enhance_path, model):
    image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path)
    origin_predict = remove_small_points(origin_path, 40)
    enhance_predict = remove_small_points(enhance_path, 40)
    true_mask = true_mask[:, :, 0] // 255 if len(true_mask.shape) == 3 else true_mask // 255
    origin_predict = origin_predict[:, :, 0] // 255 if len(origin_predict.shape) == 3 else origin_predict // 255
    enhance_predict = enhance_predict[:, :, 0] // 255 if len(enhance_predict.shape) == 3 else enhance_predict // 255
    true_mask_label, true_num = measure.label(true_mask, connectivity=2, return_num=True)
    origin_predict_label, origin_num = measure.label(origin_predict, connectivity=2, return_num=True)
    enhance_predict_label, enhance_num = measure.label(enhance_predict, connectivity=2, return_num=True)
    true_props = measure.regionprops(true_mask_label)
    origin_props = measure.regionprops(origin_predict_label)
    enhance_props = measure.regionprops(enhance_predict_label)
    # for i in range(len(true_props)):
    #     bbox = true_props[i].bbox
    #     cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 1)
    # save_name = image_path.split("\\")[-1]
    # save_name = os.path.join("../predict/test", save_name)
    # cv2.imwrite(save_name, image)
    # print(1111)
    flag_arr = np.zeros(true_num)
    origin_image = image.copy()
    origin_tp = 0
    for i in range(len(origin_props)):
        center_y, center_x = origin_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        flag = true_mask_label[center_y, center_x]
        if flag != 0 and flag_arr[flag - 1] != 1:
            flag_arr[flag - 1] = 1
            true_y, true_x = true_props[flag - 1].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(origin_image, (true_x, true_y), 5, (0, 255, 0), -1)
            origin_tp += 1
        else:
            cv2.circle(origin_image, (center_x, center_y), 5, (255, 0, 0), -1)
    origin_fp = origin_num - origin_tp
    origin_fn = true_num - origin_tp
    for i in range(len(flag_arr)):
        if flag_arr[i] == 0:
            true_y, true_x = true_props[i].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(origin_image, (true_x, true_y), 5, (0, 0, 255), -1)
    #
    # save_name = image_path.split("\\")[-1]
    # save_name = os.path.join("../predict/test", save_name)
    #
    # cv2.imwrite(save_name, origin_image)
    # 	70,130,180
    flag_arr = np.zeros(true_num)
    enhance_image = image.copy()
    enhance_tp = 0
    for i in range(len(enhance_props)):
        center_y, center_x = enhance_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        flag = true_mask_label[center_y, center_x]
        if flag != 0 and flag_arr[flag - 1] != 1:
            flag_arr[flag - 1] = 1
            enhance_tp += 1
            true_y, true_x = true_props[flag - 1].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 5, (0, 255, 0), -1)
        else:
            cv2.circle(enhance_image, (center_x, center_y), 5, (255, 0, 0), -1)

    enhance_fp = enhance_num - enhance_tp
    enhance_fn = true_num - enhance_tp

    for i in range(len(flag_arr)):
        if flag_arr[i] == 0:
            true_y, true_x = true_props[i].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 5, (0, 0, 255), -1)
    save_root_path = "../predict/test"
    save_name = enhance_path.split("\\")[-1]
    origin_save_path = os.path.join(save_root_path, "origin2", model)
    enhance_save_path = os.path.join(save_root_path, "enhance2", model)
    if not os.path.exists(origin_save_path):
        os.mkdir(origin_save_path)
    if not os.path.exists(enhance_save_path):
        os.mkdir(enhance_save_path)
    origin_save_path = os.path.join(origin_save_path, save_name)
    enhance_save_path = os.path.join(enhance_save_path, save_name)
    cv2.imwrite(origin_save_path, origin_image)
    print("origin predict image save to " + origin_save_path)
    cv2.imwrite(enhance_save_path, enhance_image)
    print("enhance predict image save to " + enhance_save_path)


def run_drawResultFigure():
    image_root_path = "../DRIVE224_3/test/images"
    mask_root_path = "../DRIVE224_3/test/masks"
    origin_root_path = "../predict/origin2/"
    enhance_root_path = "../predict/enhance2/"
    model_list = ['UNet', 'UNetPP', 'AttentionUNet', 'TransUNet', 'SwinUNet',
                  'MSFTransUNet', 'DATransUNet']
    image_list = os.listdir(image_root_path)

    for image in image_list:
        cell = image.split('_')[0]
        if cell == "A549" or cell == "MCF7":
            image_path = os.path.join(image_root_path, image)
            mask = image.split('.')[0] + "_mask.png"
            mask_path = os.path.join(mask_root_path, mask)
            predict = mask.split('.')[0] + "_predict.png"
            for model in model_list:
                origin_path = os.path.join(origin_root_path, model)
                enhance_path = os.path.join(enhance_root_path, model)
                origin_path = os.path.join(origin_path, predict)
                enhance_path = os.path.join(enhance_path, predict)
                drawResultFigure(image_path, mask_path, origin_path, enhance_path, model)


# 针对MBM
def drawResultFigureMBM(image_path, mask_path, enhance_path, model):
    image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path)
    # origin_predict = remove_small_points(origin_path, 40)
    enhance_predict = cv2.imread(enhance_path)
    true_mask = true_mask[:, :, 0] // 255 if len(true_mask.shape) == 3 else true_mask // 255
    enhance_predict = enhance_predict[:, :, 0] // 255 if len(enhance_predict.shape) == 3 else enhance_predict // 255
    true_mask_label, true_num = measure.label(true_mask, connectivity=2, return_num=True)
    enhance_predict_label, enhance_num = measure.label(enhance_predict, connectivity=2, return_num=True)
    true_props = measure.regionprops(true_mask_label)
    enhance_props = measure.regionprops(enhance_predict_label)
    # for i in range(len(true_props)):
    #     bbox = true_props[i].bbox
    #     cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 1)
    # save_name = image_path.split("\\")[-1]
    # save_name = os.path.join("../predict/test", save_name)
    # cv2.imwrite(save_name, image)
    # print(1111)
    #
    # save_name = image_path.split("\\")[-1]
    # save_name = os.path.join("../predict/test", save_name)
    #
    # cv2.imwrite(save_name, origin_image)
    # 	70,130,180
    flag_arr = np.zeros(true_num)
    enhance_image = image.copy()
    enhance_tp = 0
    for i in range(len(enhance_props)):
        center_y, center_x = enhance_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        flag = true_mask_label[center_y, center_x]
        if flag != 0 and flag_arr[flag - 1] != 1:
            flag_arr[flag - 1] = 1
            enhance_tp += 1
            true_y, true_x = true_props[flag - 1].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 3, (0, 255, 0), -1)
        # else:
        #     cv2.circle(enhance_image, (center_x, center_y), 3, (255, 0, 0), -1)

    enhance_fp = enhance_num - enhance_tp
    enhance_fn = true_num - enhance_tp

    for i in range(len(flag_arr)):
        if flag_arr[i] == 0:
            true_y, true_x = true_props[i].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 3, (0, 0, 255), -1)
    save_root_path = "../predict/"
    save_name = enhance_path.split("/")[-1]
    enhance_save_path = os.path.join(save_root_path, "enhance_MBM", model + "_counting")
    if not os.path.exists(enhance_save_path):
        os.mkdir(enhance_save_path)
    enhance_save_path = os.path.join(enhance_save_path, save_name)
    cv2.imwrite(enhance_save_path, enhance_image)
    print("enhance predict image save to " + enhance_save_path)


def run_drawResultFigureMBM():
    image_root_path = "../DRIVE224_4/test/images"
    mask_root_path = "../DRIVE224_4/test/enhanceMasks"
    enhance_root_path = "../predict/enhance_MBM/"
    model_list = ['UNet', 'SAUNet', 'CellUNet', 'CResUNet', 'SCTransUNet']
    image_list = os.listdir(image_root_path)

    for image in image_list:
        cell = image.split('_')[0]
        if cell == "A549" or cell == "MCF7" or cell == 'BMB':
            image_path = os.path.join(image_root_path, image)
            mask = image.split('.')[0] + "_mask_enhance.png"
            mask_path = os.path.join(mask_root_path, mask)
            predict = image.split('.')[0] + '_mask_predict.png'
            for model in model_list:
                enhance_path = os.path.join(enhance_root_path, model)
                enhance_path = os.path.join(enhance_path, predict)
                drawResultFigureMBM(image_path, mask_path, enhance_path, model)


# 针对MBM
def drawGTMBM(image_path, mask_path):
    image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path)
    # origin_predict = remove_small_points(origin_path, 40)

    true_mask = true_mask[:, :, 0] // 255 if len(true_mask.shape) == 3 else true_mask // 255
    true_mask_label, true_num = measure.label(true_mask, connectivity=2, return_num=True)
    true_props = measure.regionprops(true_mask_label)
    true_image = image.copy()
    enhance_tp = 0
    for i in range(len(true_props)):
        center_y, center_x = true_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        cv2.circle(true_image, (center_x, center_y), 3, (0, 255, 0), -1)
    save_root_path = "../predict/"
    save_name = image_path.split("/")[-1]
    true_save_path = os.path.join(save_root_path, "enhance_liveCell", 'GT')
    if not os.path.exists(true_save_path):
        os.mkdir(true_save_path)
    true_save_path = os.path.join(true_save_path, save_name)
    cv2.imwrite(true_save_path, true_image)
    print("enhance predict image save to " + true_save_path)


def run_drawGTMBM():
    image_root_path = "../DRIVE224_2/test/images"
    mask_root_path = "../DRIVE224_2/test/enhanceMasks"
    image_list = os.listdir(image_root_path)

    for image in image_list:
        cell = image.split('_')[0]
        if cell == "A549" or cell == "MCF7" or cell == 'BMB':
            image_path = os.path.join(image_root_path, image)
            mask = image.split('.')[0] + "_mask_enhance.png"
            mask_path = os.path.join(mask_root_path, mask)
            drawGTMBM(image_path, mask_path)


def convertDot(mask_path):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    for i in range(len(original_props)):
        # cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), 5,
        #            (255, 255, 255), -1)
        center_x = int(original_props[i].centroid[1])
        center_y = int(original_props[i].centroid[0])
        enhance_mask[center_y, center_x, :] = 255
    enhance_mask_name = '_'.join(os.path.splitext(mask_path.split("/")[-1])[0].split('_')[0:-1]) + '_dot.png'
    enhance_mask_path = '../predict/test'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


def runConvertDot():
    mask_path = "../DRIVE224_4/training/enhanceMasks/BMB_5_3_mask_enhance.png"
    convertDot(mask_path)


def drawHistogram():
    # Sample data
    categories = ['mIoU', 'dice']
    # categories = ['accuracy', 'precise', 'recall', 'f1']
    group1 = [77.08, 81.31]
    group2 = [77.54, 81.56]
    group3 = [77.50, 81.48]
    group4 = [77.65, 81.71]
    group5 = [77.81, 81.83]

    # group1 = [38.54, 70.54, 45.70, 54.26]
    # group2 = [42.20, 72.77, 49.85, 58.16]
    # group3 = [42.75, 73.28, 50.33, 58.68]
    # group4 = [44.37, 73.48, 55.83, 62.31]
    # group5 = [46.52, 74.21, 55.06, 63.35]

    # group1 = [86.48, 88.73]
    # group2 = [86.74, 89.16]
    # group3 = [86.62, 88.98]
    # group4 = [86.85, 89.23]
    # group5 = [87.19, 89.52]

    # group1 = [53.03, 68.34, 69.72, 67.00]
    # group2 = [54.62, 69.48, 70.67, 68.95]
    # group3 = [54.95, 69.57, 71.03, 69.51]
    # group4 = [55.70, 69.04, 72.92, 69.75]
    # group5 = [57.80, 72.50, 72.39, 71.06]

    min_value = min(group1 + group2 + group3 + group4 + group5) - 1
    x = np.arange(len(categories))  # the label locations
    width = 0.18  # the width of the bars
    fig, ax = plt.subplots()

    # Plot group1, group2, group3, group4, and group5 as histograms
    rects1 = ax.bar(x - 2 * width, group1, width, label='Baseline', color="#8ECFC9")
    rects2 = ax.bar(x - width, group2, width, label='B+GC-T', color="#FFBE7A")
    rects3 = ax.bar(x, group3, width, label='B+GC-T+MK-C', color="#FA7F6F")
    rects4 = ax.bar(x + width, group4, width, label='B+GC-T+MK-C+BN', color="#BEB8DC")
    rects5 = ax.bar(x + 2 * width, group5, width, label='MSF-TransUNet', color="#82B0D2")

    # Calculate the x positions for the line charts to be in the middle of the bars
    line_x = x  # Use the same x positions for simplicity

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=min_value)
    fig.tight_layout()
    plt.savefig("../predict/fig/livecell2.tif")
    plt.show()

    def drawHistogram():
        # Sample data
        categories = ['mIoU', 'dice']
        # categories = ['accuracy', 'precise', 'recall', 'f1']
        group1 = [77.08, 81.31]
        group2 = [77.54, 81.56]
        group3 = [77.50, 81.48]
        group4 = [77.65, 81.71]
        group5 = [77.81, 81.83]

        # group1 = [38.54, 70.54, 45.70, 54.26]
        # group2 = [42.20, 72.77, 49.85, 58.16]
        # group3 = [42.75, 73.28, 50.33, 58.68]
        # group4 = [44.37, 73.48, 55.83, 62.31]
        # group5 = [46.52, 74.21, 55.06, 63.35]

        # group1 = [86.48, 88.73]
        # group2 = [86.74, 89.16]
        # group3 = [86.62, 88.98]
        # group4 = [86.85, 89.23]
        # group5 = [87.19, 89.52]

        # group1 = [53.03, 68.34, 69.72, 67.00]
        # group2 = [54.62, 69.48, 70.67, 68.95]
        # group3 = [54.95, 69.57, 71.03, 69.51]
        # group4 = [55.70, 69.04, 72.92, 69.75]
        # group5 = [57.80, 72.50, 72.39, 71.06]

        min_value = min(group1 + group2 + group3 + group4 + group5) - 1
        x = np.arange(len(categories))  # the label locations
        width = 0.18  # the width of the bars
        fig, ax = plt.subplots()

        # Plot group1, group2, group3, group4, and group5 as histograms
        rects1 = ax.bar(x - 2 * width, group1, width, label='Baseline', color="#8ECFC9")
        rects2 = ax.bar(x - width, group2, width, label='B+GC-T', color="#FFBE7A")
        rects3 = ax.bar(x, group3, width, label='B+GC-T+MK-C', color="#FA7F6F")
        rects4 = ax.bar(x + width, group4, width, label='B+GC-T+MK-C+BN', color="#BEB8DC")
        rects5 = ax.bar(x + 2 * width, group5, width, label='MSF-TransUNet', color="#82B0D2")

        # Calculate the x positions for the line charts to be in the middle of the bars
        line_x = x  # Use the same x positions for simplicity

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=min_value)
        fig.tight_layout()
        plt.savefig("../predict/fig/livecell2.tif")
        plt.show()


def drawHistogram2():
    # 模型名称
    import matplotlib.pyplot as plt

    # 模型名称
    models = ['U-Net', 'U-Net++', 'Attention-UNet', 'TransUNet', 'Swin-UNet', 'DA-TransUNet', 'MSF-TransUNet']

    # 对应的准确率提升值（示例数据）
    # accuracy_improvement1 = [38.54, 32.49, 44.95, 42.29, 21.56, 45.69, 46.52]
    # accuracy_improvement2 = [59.51, 52.03, 60.33, 60.54, 49.71, 60.86, 61.19]

    accuracy_improvement1 = [53.03, 41.83, 55.15, 49.50, 42.38, 55.59, 57.80]
    accuracy_improvement2 = [63.55, 56.45, 64.97, 58.71, 56.80, 65.66, 69.07]

    accuracy_improvements = []
    for i in range(len(accuracy_improvement1)):
        accuracy_improvements.append(accuracy_improvement2[i] - accuracy_improvement1[i])
    # 使用和谐的颜色
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

    # 绘制柱状图
    plt.figure(figsize=(10, 6), dpi=600)
    bars = plt.bar(models, accuracy_improvements, color=colors)

    plt.ylim(5, 20.1)

    # 添加标题和标签
    # plt.title('Accuracy improvement of different models on A549', fontsize=16)
    plt.ylabel('Accuracy improvement (%)', fontsize=16)

    # 将模型名称水平显示
    plt.xticks(rotation=15, fontsize=16)
    plt.yticks(fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, round(yval, 2),
                 ha='center', va='bottom', fontsize=15)
    plt.yticks(np.arange(5, 20.1, 3))
    # 显示柱状图
    plt.tight_layout()
    # plt.show()

    plt.savefig("../predict/fig/compare2.png")


def drawLineGraph():
    # 细胞密度 (10% 到 70%)
    cell_density = [10, 20, 30, 40, 50, 60, 70]

    # 对应的准确率、精确率、召回率和 F1 分数（示例数据）
    accuracy = [85.75, 78.94, 72.26, 68.34, 64.58, 60.42, 55.64]
    precision = [94.35, 89.53, 85.37, 81.05, 77.43, 74.16, 68.08]
    recall = [88.69, 84.48, 80.74, 77.58, 75.62, 73.18, 70.19]
    f1_score = [88.93, 85.38, 82.35, 78.50, 75.97, 72.86, 68.86]

    # 提高图像清晰度并设置图形大小，dpi 设置为 600
    plt.figure(figsize=(8, 6), dpi=600)

    # 使用期刊审美颜色：灰度配色或中性色
    plt.plot(cell_density, accuracy, marker='o', linestyle='-', color='#1f77b4', label='Accuracy')  # 蓝灰色
    plt.plot(cell_density, precision, marker='s', linestyle='--', color='#ff7f0e', label='Precision')  # 柔和的橙色
    plt.plot(cell_density, recall, marker='^', linestyle='-.', color='#2ca02c', label='Recall')  # 深绿色
    plt.plot(cell_density, f1_score, marker='d', linestyle=':', color='#d62728', label='F1 Score')  # 深红色

    # 设置标题和坐标轴标签
    # plt.title('Counting Performance of the Model at Different Cell Densities', fontsize=15,
    #           fontfamily='Times New Roman')
    plt.xlabel('细胞密度 (%)', fontsize=13, fontfamily='SimSun')
    plt.ylabel('比率 (%)', fontsize=13, fontfamily='SimSun')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 显示网格线
    # plt.grid(True)

    # 显示图例，期刊通常将图例放置在图形的上方或旁边
    plt.legend(loc='best')

    # 展示图表
    plt.tight_layout()
    # plt.show()
    plt.savefig("../predict/fig/density2.png")


# 绘制不同模型的计数结果图
def drawLineGraph2():
    # 细胞密度 (10% 到 70%)
    models = ['Swin-UNet', 'UNet++', 'UNet', 'Attention-UNet', 'TransUNet', 'DA-TransUNet', 'MSF-TransUNet']

    # 对应的准确率、精确率、召回率和 F1 分数（示例数据）
    accuracy = [49.71, 52.03, 60.86, 60.33, 60.54, 59.51, 61.35]
    precision = [63.32, 72.01, 75.34, 73.86, 75.23, 73.92, 77.73]
    recall = [68.87, 64.51, 75.26, 75.86, 74.90, 74.43, 74.32]
    f1_score = [65.41, 67.51, 74.89, 74.33, 74.52, 73.67, 75.48]

    # accuracy = [56.80, 56.45, 63.55, 64.97, 58.71, 65.66, 70.21]
    # precision = [65.47, 71.51, 76.51, 76.69, 69.40, 77.27, 83.86]
    # recall = [73.28, 70.64, 76.81, 77.92, 76.49, 79.32, 79.24]
    # f1_score = [70.73, 70.07, 76.09, 76.71, 72.12, 77.65, 80.93]

    # 提高图像清晰度并设置图形大小，dpi 设置为 600
    plt.figure(figsize=(8, 6), dpi=600)

    # 使用期刊审美颜色：灰度配色或中性色
    plt.plot(models, accuracy, marker='o', linestyle='-', color='#1f77b4', label='Accuracy')  # 蓝灰色
    plt.plot(models, precision, marker='s', linestyle='--', color='#ff7f0e', label='Precision')  # 柔和的橙色
    plt.plot(models, recall, marker='^', linestyle='-.', color='#2ca02c', label='Recall')  # 深绿色
    plt.plot(models, f1_score, marker='d', linestyle=':', color='#d62728', label='F1 Score')  # 深红色

    # 设置标题和坐标轴标签
    plt.title('Experimental results of different models on LiveCell dataset', fontsize=15,
              fontfamily='Times New Roman')
    # plt.xlabel('Cell Density (%)', fontsize=13, fontfamily='Times New Roman')
    # plt.ylabel('Metrics (%)', fontsize=14, fontfamily='Times New Roman')

    plt.xticks(rotation=15, fontsize=12)
    plt.yticks(fontsize=12)

    # 显示网格线
    # plt.grid(True)

    # 显示图例，期刊通常将图例放置在图形的上方或旁边
    plt.legend(loc='best')

    # 展示图表
    plt.tight_layout()
    # plt.show()
    plt.savefig("../predict/fig/result1.png")


def data_annotation_enhance_radius(mask_path):
    original_mask = cv2.imread(mask_path)
    original_mask_copy = original_mask
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    cell_name = mask_path.split('\\')[-1].split('_')[0]
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    areas = []
    for i in range(len(original_props)):
        areas.append(original_props[i].area)
    sort_areas = sorted(areas)
    threshold = sort_areas[original_num // 2]
    for i in range(len(original_props)):
        if original_props[i].area > threshold:
            if cell_name == 'A549':
                radius = 15
            else:
                radius = 7
        else:
            if cell_name == "A549":
                # if original_props[i].area <= 50:
                #     radius = 5
                # else:
                #     radius = 7
                radius = 5
            else:
                # if original_props[i].area <= 15:
                #     radius = 2
                # else:
                #     radius = 3
                radius = 3
        # cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), radius,
        #            (255, 255, 255), -1)
        cv2.circle(original_mask_copy, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])),
                   radius, (0, 205, 102), -1)
    enhance_mask_name = mask_path.split('\\')[-1].split('.')[0] + '_enhance_radius_15.png'
    enhance_mask_path = '../predict/enhance_mask_radius/'+cell_name
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, original_mask_copy)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


def run_data_annotation_enhance_radius():
    mask_root_path = '../DRIVE224_3/'
    flags = ["training", "val", "test"]
    for name in flags:
        mask_path = os.path.join(mask_root_path, name, "masks")
        image_list = os.listdir(mask_path)
        for mask_name in image_list:
            cell = mask_name.split('_')[0]
            if cell == "MCF7" or cell == "A549":
                msk_path = os.path.join(mask_path, mask_name)
                data_annotation_enhance_radius(msk_path)
            else:
                continue




if __name__ == "__main__":
    # convertAnn("../predict/test/mask.png")
    # test_DCN()
    # run_remove_small_points()
    # run_data_annotation_enhance()
    # run_postprocess()
    # compute_mean_std("../DRIVE224_3/training/images")
    # run_drawResultFigure()
    # drawHistogram()
    # run_data_annotation_enhance_BM()
    # runConvertDot()
    # run_drawResultFigureMBM()
    # run_drawGTMBM()
    # runConvertDot()
    # drawLineGraph()
    # drawHistogram2()
    # drawLineGraph()
    # run_data_annotation_enhance_radius()
    drawHistogram2()
