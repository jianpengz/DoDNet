import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label as LAB
from skimage.transform import resize
import SimpleITK as sitk
import argparse

from medpy.metric import hd95

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dynconv post processing!")

    parser.add_argument("--img_folder_path", type=str, default='outputs/dodnet/')

    return parser.parse_args()

args = get_arguments()

def continues_region_extract_organ(label, keep_region_nums):  # keep_region_nums=1
    mask = False*np.zeros_like(label)
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    #
    ary_num = np.zeros(shape=(n+1,1))
    for i in range(0, n+1):
        ary_num[i] = np.sum(L==i)
    max_index = np.argsort(-ary_num, axis=0)
    count=1
    for i in range(1, n+1):
        if count<=keep_region_nums: # keep
            mask = np.where(L == max_index[i][0], label, mask)
            count+=1
    label = np.where(mask==True, label, np.zeros_like(label))
    return label

def continues_region_extract_tumor(label):  #

    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    for i in range(1, n+1):
        if np.sum(L==i)<=50 and n>1: # remove default 50
            label = np.where(L == i, np.zeros_like(label), label)

    return label



def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1

    dice = 2 * num / den

    return dice.mean()


def task_index(name):
    if "liver" in name:
        return 0
    if "case" in name:
        return 1
    if "hepa" in name:
        return 2
    if "pancreas" in name:
        return 3
    if "colon" in name:
        return 4
    if "lung" in name:
        return 5
    if "spleen" in name:
        return 6

def compute_HD95(ref, pred):
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))

val_Dice = np.zeros(shape=(7, 2))
val_HD = np.zeros(shape=(7, 2))
count = np.zeros(shape=(7, 2))

for root, dirs, files in os.walk(args.img_folder_path):
    for i in sorted(files):
        if i[-12:-7] != 'label':
            continue
        i2 = i[:-12]+'pred'+i[-7:]
        i_file = root + i
        i2_file = root + i2
        predNII = nib.load(i2_file)
        labelNII = nib.load(i_file)
        pred = predNII.get_data()
        label = labelNII.get_data()

        # post-processing

        task_id = task_index(i)
        if task_id == 0 or task_id == 1 or task_id == 3:
            pred_organ = (pred >= 1)
            pred_tumor = (pred == 2)
            label_organ = (label >= 1)
            label_tumor = (label == 2)

        elif task_id == 2:
            pred_organ = (pred == 1)
            pred_tumor = (pred == 2)
            label_organ = (label == 1)
            label_tumor = (label == 2)

        elif task_id == 4 or task_id == 5:
            pred_organ = None
            pred_tumor = (pred == 2)
            label_organ = None
            label_tumor = (label == 2)
        elif task_id == 6:
            pred_organ = (pred == 1)
            pred_tumor = None
            label_organ = (label == 1)
            label_tumor = None
        else:
            print("No such a task!")

        if task_id == 0:
            pred_organ = continues_region_extract_organ(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_tumor(pred_tumor)
        elif task_id == 1:
            pred_organ = continues_region_extract_organ(pred_organ, 2)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
        elif task_id == 2:
            pred_tumor = continues_region_extract_tumor(pred_tumor)
        elif task_id == 3: 
            pred_organ = continues_region_extract_organ(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_tumor(pred_tumor)
        elif task_id == 4: 
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
        elif task_id == 5:
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
        elif task_id == 6:
            pred_organ = continues_region_extract_organ(pred_organ, 1)
        else:
            print("No such a task index!!!")

        if label_organ is not None:
            dice_c1 = dice_score(pred_organ, label_organ)
            HD_c1 = compute_HD95(label_organ, pred_organ)
            val_Dice[task_id, 0] += dice_c1
            val_HD[task_id, 0] += HD_c1
            count[task_id, 0] += 1
        else:
            dice_c1=-999
            HD_c1=999
        if label_tumor is not None:
            dice_c2 = dice_score(pred_tumor, label_tumor)
            HD_c2 = compute_HD95(label_tumor, pred_tumor)
            val_Dice[task_id, 1] += dice_c2
            val_HD[task_id, 1] += HD_c2
            count[task_id, 1] += 1
        else:
            dice_c2=-999
            HD_c2=999
        print("%s: Organ_Dice %f, tumor_Dice %f | Organ_HD %f, tumor_HD %f" % (i[:-13], dice_c1, dice_c2, HD_c1, HD_c2))

count[count == 0] = 1
val_Dice = val_Dice / count
val_HD = val_HD / count

print("Sum results")
for t in range(7):
    print('Sum: Task%d- Organ_Dice:%.4f Tumor_Dice:%.4f | Organ_HD:%.4f Tumor_HD:%.4f' % (t, val_Dice[t, 0], val_Dice[t, 1], val_HD[t,0], val_HD[t,1]))
