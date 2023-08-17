import argparse
import os, sys

sys.path.append("..")

import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from models.TransDoDNet import MOTS_DeformTR as MOTS_model
from MOTSDataset import MOTSTestDataSet, my_collate
import timeit
from utils.ParaFlop import print_model_parm_nums
import nibabel as nib
from math import ceil
import math
from engine import Engine
from apex import amp
from skimage.measure import label as LAB
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from medpy.metric import hd95, asd
from collections import OrderedDict
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MOTS: Single solution!")

    parser.add_argument("--data_dir", type=str, default='../data_list/')
    parser.add_argument("--val_list", type=str, default='MOTS/MOTS_test.txt') #0Liver_val5f_1
    parser.add_argument("--reload_path", type=str, default='')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--nnUNet_preprocessed", type=str, default=os.environ['nnUNet_preprocessed'])
    parser.add_argument("--save_path", type=str, default='')

    parser.add_argument("--input_size", type=str, default='64,192,192')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # Res
    parser.add_argument('--res_depth', default=50, type=int)
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8')


    # * Transformer
    parser.add_argument("--using_transformer", type=str2bool, default=True)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=768, type=int)
    parser.add_argument('--hidden_dim', default=192, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=7, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_feature_levels', default=3, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--normalize_before', default=False, type=str2bool)
    parser.add_argument('--deepnorm', default=True, type=str2bool)
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument("--add_memory", type=int, default=2, choices=(0,1,2)) # feature fusion: 0: cnn; 1:tr; 2:cnn+tr

    parser.add_argument("--isSD", type=str2bool, default=True)
    parser.add_argument("--isHD", type=str2bool, default=True)
    parser.add_argument("--ifGa", type=str2bool, default=True)

    return parser


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003



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

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result


def compute_dice_score(preds, labels):
    # preds: 1x4x128x128x128
    # labels: 1x128x128x128

    preds = torch.sigmoid(preds)

    pred_pa = preds[:, 0, :, :, :]
    label_pa = labels[:, 0, :, :, :]
    dice_pa = dice_score(pred_pa, label_pa)

    return dice_pa

def compute_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 999.
    elif num_pred == 0 and num_ref != 0:
        return 999.
    else:
        return hd95(pred, ref, (1, 1, 1))

def compute_SD(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 999.
    elif num_pred == 0 and num_ref != 0:
        return 999.
    else:
        return asd(pred, ref, (1, 1, 1))

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()
    with torch.no_grad():
        padded_prediction = net_list[0](img, task_id)
        padded_prediction = torch.sigmoid(padded_prediction)
        for i in range(1, len(net_list)):
            padded_prediction_i = net_list[i](img, task_id)
            padded_prediction_i = torch.sigmoid(padded_prediction_i)
            padded_prediction += padded_prediction_i
        padded_prediction /= len(net_list)
    return padded_prediction#.cpu()  # .cpu().data.numpy()


def predict_sliding(args, net_list, image, tile_size, classes, task_id, gaussian_importance_map):  # tile_size:32x256x256
    # padding or not?
    flag_padding = False
    dept_missing = math.ceil(tile_size[0] - image.shape[2])
    rows_missing = math.ceil(tile_size[1] - image.shape[3])
    cols_missing = math.ceil(tile_size[2] - image.shape[4])
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0
    if dept_missing < 0:
        dept_missing = 0
    image = np.pad(image, ((0, 0), (0, 0), (0, dept_missing), (0, rows_missing), (0, cols_missing)), constant_values = (-1,-1))

    image_size = image.shape
    overlap = 0.5
    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros(
        (image_size[0], args.num_queries, classes, image_size[2], image_size[3], image_size[4]))  
    count_predictions = np.zeros(
        (image_size[0], args.num_queries, classes, image_size[2], image_size[3], image_size[4])) 
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in tqdm(range(tile_deps)):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  
                y1 = max(int(y2 - tile_size[1]), 0)  

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                prediction1 = multi_net(net_list, img, task_id)
                prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), task_id), [2+1])
                prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), task_id), [3+1])
                prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), task_id), [4+1])
                prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3]), task_id), [2+1, 3+1])
                prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2, 4]), task_id), [2+1, 4+1])
                prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3, 4]), task_id), [3+1, 4+1])
                prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3, 4]), task_id), [2+1, 3+1, 4+1])
                prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                prediction = prediction.cpu()
                if args.FP16:
                    prediction = prediction.to(torch.float64)
                if args.ifGa:
                    prediction = prediction * gaussian_importance_map
                    count_predictions[:, :, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                else:
                    count_predictions[:, :, :, d1:d2, y1:y2, x1:x2] += 1

                full_probs[:, :, :, d1:d2, y1:y2, x1:x2] += prediction
                

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    
    return full_probs[:,:,:,:(image_size[2]-dept_missing), :(image_size[3]-rows_missing), :(image_size[4]-cols_missing)]


def save_nii(args, pred, name, properties):  # bs, c, WHD

    segmentation = pred

    # save
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    current_shape = segmentation.shape
    shape_original_after_cropping = np.array(torch.as_tensor(properties.get('size_after_cropping')), dtype=int)[[0,2,1]]
    shape_original_before_cropping = properties.get('original_size_of_raw_data')[0].data.numpy()[[0,2,1]]

    order = 0
    force_separate_z = None

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_original_after_cropping, 0, 0)
        else:
            if force_separate_z is None:
                if get_do_separate_z(properties.get('original_spacing').data.numpy()[0]):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                elif get_do_separate_z(properties.get('spacing_after_resampling').data.numpy()):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('spacing_after_resampling').data.numpy()[0])
                else:
                    do_separate_z = False
                    lowres_axis = None
            else:
                do_separate_z = force_separate_z
                if do_separate_z:
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                else:
                    lowres_axis = None

            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
            seg_old_spacing = resample_data_or_seg(segmentation[None], shape_original_after_cropping, is_seg=True,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                                   order_z=0)[0]
    else:
        seg_old_spacing = segmentation

    bbox = properties.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if name[0][:4] == 'kidn' or name[0][:4] == 'case':
        seg_old_size = np.rot90(seg_old_size[:, ::-1, :], 1, [1, 2])
        seg_old_size = seg_old_size.transpose([1, 2, 0])
        name[0] = name[0].replace("kidney","case")

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(np.array(torch.as_tensor(properties['itk_spacing'])).astype(np.float64))
    seg_resized_itk.SetOrigin(np.array(torch.as_tensor(properties['itk_origin'])).astype(np.float64))
    seg_resized_itk.SetDirection(np.array(torch.as_tensor(properties['itk_direction'])).astype(np.float64))
    sitk.WriteImage(seg_resized_itk, args.save_path + '/' + name[0]+'.nii.gz')

    return None


def extract_continues_region(label, keep_region_nums):  # keep_region_nums=1
    mask = False*np.zeros_like(label)
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

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
        # else: # remove
        #     label = np.where(L == max_index[i][0], np.zeros_like(label), label)
    label = np.where(mask==True, label, np.zeros_like(label))
    return label

def continues_region_extract_tumor(label):  #

    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    #
    for i in range(1, n+1):
        print("sum_t_%d" % (np.sum(L==i)))
        if np.sum(L==i)<=50: # remove  50:66.26 #30:66.31
            label = np.where(L == i, np.zeros_like(label), label)

    return label

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

def validate(args, input_size, model, ValLoader, num_classes, engine, json_dict):
    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, task_id, properties = batch
        print("Processing %s (%s)" % (name[0], image.shape))
        # image = image.cuda()
        # label = label.cuda()
        # image = image[:, :, 130:130 + 64, 100:100 + 128, 100:100 + 128]
        # label = label[:, :, 130:130 + 64, 100:100 + 128, 100:100 + 128]
        
        gaussian_importance_map = _get_gaussian(input_size, sigma_scale=1. / 8)
        pred = predict_sliding(args, model, image.numpy(), input_size, num_classes, task_id, gaussian_importance_map)
        size_after_resampling = np.array(properties["size_after_resampling"])
        pred = pred[:,:,:,0:size_after_resampling[0], 0:size_after_resampling[1], 0:size_after_resampling[2]]

        if args.using_transformer:
            N_b, N_q, N_c, N_d, N_h, N_w = pred.shape
            pred_convert = torch.zeros(size=(N_b, N_c, N_d, N_h, N_w)).cuda()

            for i_b in range(N_b):
                pred_convert[i_b] = pred[i_b, int(task_id[i_b])]
        else:
            pred_convert = pred
        pred_convert = pred_convert.cpu().data.numpy()

        seg_pred_2class = np.asarray(np.around(pred_convert), dtype=np.uint8)
        # seg_pred_2class = np.where(pred>=0.6, np.ones_like(pred), np.zeros_like(pred))
        pred_organ = seg_pred_2class[0, 0, :, :, :] # bs should be 1
        pred_tumor = seg_pred_2class[0, 1, :, :, :]
        seg_pred = np.zeros_like(pred_organ)
        
        if name[0][:4] == 'live':
            pred_organ = extract_continues_region(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_tumor(pred_tumor)
            pred_all = np.where(pred_organ == 1, 1, seg_pred)
            pred_all = np.where(pred_tumor == 1, 2, pred_all)
        elif name[0][:4] == 'kidn':
            pred_organ = extract_continues_region(pred_organ, 2)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = extract_continues_region(pred_tumor, 1)
            pred_all = np.where(pred_organ == 1, 1, seg_pred)
            pred_all = np.where(pred_tumor == 1, 2, pred_all)
        elif name[0][:4] == 'hepa':
            pred_liver = pred[0, 0, 0]
            pred_liver = np.asarray(np.around(pred_liver), dtype=np.uint8)
            pred_liver = extract_continues_region(pred_liver, 1)
            pred_tumor = np.where(pred_liver == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_all = np.where(pred_organ == 1, 1, seg_pred)
            pred_all = np.where(pred_tumor == 1, 2, pred_all)
        elif name[0][:4] == 'panc':
            pred_organ = extract_continues_region(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_all = np.where(pred_organ == 1, 1, seg_pred)
            pred_all = np.where(pred_tumor == 1, 2, pred_all)
        elif name[0][:4] == 'colo':
            pred_tumor = extract_continues_region(pred_tumor, 1)
            pred_all = np.where(pred_tumor == 1, 1, seg_pred)
        elif name[0][:4] == 'lung':
            pred_tumor = extract_continues_region(pred_tumor, 1)
            pred_all = np.where(pred_tumor == 1, 1, seg_pred)
        elif name[0][:4] == 'sple':
            pred_organ[:,:,250:]=0
            pred_organ = extract_continues_region(pred_organ, 1)
            pred_all = np.where(pred_organ == 1, 1, seg_pred)
        else:
            print("!!!Cannot find the task in evaluate_Single.py!!!")

        # save
        save_nii(args, pred_all, name, properties)
        print("Saving done.")


    # evaluate metrics
    print("Start to evaluate...")
    val_Dice = torch.zeros(size=(7, 2))  # np.zeros(shape=(7, 2))
    val_HD = torch.zeros(size=(7, 2))
    val_SD = torch.zeros(size=(7, 2))
    count_Dice = torch.zeros(size=(7, 2))  # np.zeros(shape=(7, 2))
    count_HD = torch.zeros(size=(7, 2))  # np.zeros(shape=(7, 2))
    count_SD = torch.zeros(size=(7, 2))  # np.zeros(shape=(7, 2))

    for root, dirs, files in os.walk(args.save_path):
        for i in sorted(files):
            if i[-6:]!='nii.gz':
                continue
            i_file = os.path.join(root, i)
            i2_file = os.path.join(os.environ["nnUNet_preprocessed"], 'Task100_MOTS/gt_segmentations', i)
            predNII = nib.load(i_file)
            labelNII = nib.load(i2_file)
            pred = predNII.get_data()
            label = labelNII.get_data()
            task_id = task_index(i)
            dice_1 = 0.
            dice_2 = 0.
            HD_1 = 999.
            HD_2 = 999.
            #
            if task_id==0 or task_id==1 or task_id==3:
                dice_1 = dice_score(pred>=1, label>=1)
                dice_2 = dice_score(pred==2, label==2)
                val_Dice[task_id, 0] += dice_1
                val_Dice[task_id, 1] += dice_2
                if args.isHD:
                    HD_1 = compute_HD95(pred >= 1, label >= 1)
                    HD_2 = compute_HD95(pred == 2, label == 2)
                else:
                    HD_1=999.
                    HD_2=999.
                if args.isSD:
                    SD_1 = compute_SD(pred >= 1, label >= 1)
                    SD_2 = compute_SD(pred == 2, label == 2)
                else:
                    SD_1=999.
                    SD_2=999.
                val_HD[task_id, 0] += HD_1 if HD_1!=999. else 0
                val_HD[task_id, 1] += HD_2 if HD_2!=999. else 0
                val_SD[task_id, 0] += SD_1 if SD_1!=999. else 0
                val_SD[task_id, 1] += SD_2 if SD_2!=999. else 0

                count_Dice[task_id, 0] += 1
                count_Dice[task_id, 1] += 1
                count_HD[task_id, 0] += 1 if HD_1!=999. else 0
                count_HD[task_id, 1] += 1 if HD_2!=999. else 0
                count_SD[task_id, 0] += 1 if SD_1!=999. else 0
                count_SD[task_id, 1] += 1 if SD_2!=999. else 0
            if task_id==2:
                dice_1 = dice_score(pred==1, label==1)
                dice_2 = dice_score(pred==2, label==2)
                val_Dice[task_id, 0] += dice_1
                val_Dice[task_id, 1] += dice_2
                if args.isHD:
                    HD_1 = compute_HD95(pred == 1, label == 1)
                    HD_2 = compute_HD95(pred == 2, label == 2)
                else:
                    HD_1=999.
                    HD_2=999.

                if args.isSD:
                    SD_1 = compute_SD(pred == 1, label == 1)
                    SD_2 = compute_SD(pred == 2, label == 2)
                else:
                    SD_1=999.
                    SD_2=999.

                val_HD[task_id, 0] += HD_1 if HD_1!=999. else 0
                val_HD[task_id, 1] += HD_2 if HD_2!=999. else 0
                val_SD[task_id, 0] += SD_1 if SD_1!=999. else 0
                val_SD[task_id, 1] += SD_2 if SD_2!=999. else 0

                count_Dice[task_id, 0] += 1
                count_Dice[task_id, 1] += 1
                count_HD[task_id, 0] += 1 if HD_1!=999. else 0
                count_HD[task_id, 1] += 1 if HD_2!=999. else 0
                count_SD[task_id, 0] += 1 if SD_1!=999. else 0
                count_SD[task_id, 1] += 1 if SD_2!=999. else 0

            elif task_id==6:
                dice_1 = dice_score(pred == 1, label == 1)
                if args.isHD:
                    HD_1 = compute_HD95(pred == 1, label == 1)
                else:
                    HD_1=999.

                if args.isSD:
                    SD_1 = compute_SD(pred == 1, label == 1)
                else:
                    SD_1=999.

                val_Dice[task_id, 0] += dice_1
                val_HD[task_id, 0] += HD_1 if HD_1!=999. else 0
                val_SD[task_id, 0] += SD_1 if SD_1!=999. else 0

                count_Dice[task_id, 0] += 1
                count_HD[task_id, 0] += 1 if HD_1!=999. else 0
                count_SD[task_id, 0] += 1 if SD_1!=999. else 0

            elif task_id==4 or task_id==5:
                dice_2 = dice_score(pred == 1, label == 1)
                if args.isHD:
                    HD_2 = compute_HD95(pred == 1, label == 1)
                else:
                    HD_2=999.

                if args.isSD:
                    SD_2 = compute_SD(pred == 1, label == 1)
                else:
                    SD_2=999.

                val_Dice[task_id, 1] += dice_2
                val_HD[task_id, 1] += HD_2 if HD_2!=999. else 0
                val_SD[task_id, 1] += SD_2 if SD_2!=999. else 0

                count_Dice[task_id, 1] += 1
                count_HD[task_id, 1] += 1 if HD_2!=999. else 0
                count_SD[task_id, 1] += 1 if SD_2!=999. else 0

            log_i = ("Organ-[Dice-%.4f; HD-%.4f; SD-%.4f], tumor-[Dice-%.4f; HD-%.4f; SD-%.4f]" % (dice_1, HD_1, SD_1, dice_2, HD_2, SD_2))
            print("%s: %s" % (i,log_i))
            json_dict[i]=log_i


    count_Dice[count_Dice == 0] = 1
    count_HD[count_HD == 0] = 1
    count_SD[count_SD == 0] = 1

    val_Dice = val_Dice / count_Dice
    val_HD = val_HD / count_HD
    val_SD = val_SD / count_SD

    print("Sum results")
    for t in range(7):
        print('Sum: Task%d- Organ:[Dice-%.4f; HD-%.4f; SD-%.4f] Tumor:[Dice-%.4f; HD-%.4f; SD-%.4f]' %
              (t, val_Dice[t, 0], val_HD[t, 0], val_SD[t, 0], val_Dice[t, 1], val_HD[t, 1], val_SD[t, 1]))

    return val_Dice.data.numpy(), val_HD.data.numpy(), val_SD.data.numpy()


def main():
    start = timeit.default_timer()
    parser = get_arguments()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        dep, wid = map(int, args.dyn_head_dep_wid.split(','))
        dyn_head_dep_wid = (dep, wid)

        cudnn.benchmark = True
        seed = 1234
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = MOTS_model(args, norm_cfg = 'IN', activation_cfg = 'LeakyReLU', num_classes = args.num_classes,
                              weight_std = args.weight_std, deep_supervision = False, res_depth=args.res_depth, dyn_head_dep_wid=dyn_head_dep_wid)

        print_model_parm_nums(model)


        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if args.FP16:
            print("Note: Using FP16 for evaluation************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    pre_dict = torch.load(args.reload_path, map_location=torch.device('cpu'))['state_dict']
                    pre_dict = {k.replace("module.", ""): v for k, v in pre_dict.items()}
                    model.load_state_dict(pre_dict)
                else:
                    pre_dict = torch.load(args.reload_path, map_location=torch.device('cpu'))['state_dict']
                    pre_dict = {k.replace("module.", ""): v for k, v in pre_dict.items()}
                    model.load_state_dict(pre_dict)
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        valloader, val_sampler = engine.get_test_loader(
            MOTSTestDataSet(args.data_dir, args.val_list))

        json_dict = OrderedDict()
        json_dict['name'] = "Single"
        json_dict["meanDice"] = OrderedDict()
        json_dict["meanHD"] = OrderedDict()
        json_dict["meanSD"] = OrderedDict()

        print('validate ...')
        model.eval()
        val_Dice, val_HD, val_SD = validate(args, input_size, [model], valloader, args.num_classes, engine, json_dict)

        json_dict["meanDice"]["liver"] = str(val_Dice[0][0])
        json_dict["meanDice"]["liver_tumor"] = str(val_Dice[0][1])
        json_dict["meanDice"]["kidney"] = str(val_Dice[1][0])
        json_dict["meanDice"]["kidney_tumor"] = str(val_Dice[1][1])
        json_dict["meanDice"]["hepatic_vessel"] = str(val_Dice[2][0])
        json_dict["meanDice"]["hepatic_vessel_tumor"] = str(val_Dice[2][1])
        json_dict["meanDice"]["pancreas"] = str(val_Dice[3][0])
        json_dict["meanDice"]["pancreas_tumor"] = str(val_Dice[3][1])
        json_dict["meanDice"]["colon"] = str(val_Dice[4][1])
        json_dict["meanDice"]["lung"] = str(val_Dice[5][1])
        json_dict["meanDice"]["spleen"] = str(val_Dice[6][0])
        json_dict["meanDice"]["AVG"] = str((val_Dice[0][0] + val_Dice[0][1] + \
                                     val_Dice[1][0] + val_Dice[1][1] + \
                                     val_Dice[2][0] + val_Dice[2][1] + \
                                     val_Dice[3][0] + val_Dice[3][1] + \
                                     val_Dice[4][1] + \
                                     val_Dice[5][1] + \
                                     val_Dice[6][0]) / 11.)

        json_dict["meanHD"]["liver"] = str(val_HD[0][0])
        json_dict["meanHD"]["liver_tumor"] = str(val_HD[0][1])
        json_dict["meanHD"]["kidney"] = str(val_HD[1][0])
        json_dict["meanHD"]["kidney_tumor"] = str(val_HD[1][1])
        json_dict["meanHD"]["hepatic_vessel"] = str(val_HD[2][0])
        json_dict["meanHD"]["hepatic_vessel_tumor"] = str(val_HD[2][1])
        json_dict["meanHD"]["pancreas"] = str(val_HD[3][0])
        json_dict["meanHD"]["pancreas_tumor"] = str(val_HD[3][1])
        json_dict["meanHD"]["colon"] = str(val_HD[4][1])
        json_dict["meanHD"]["lung"] = str(val_HD[5][1])
        json_dict["meanHD"]["spleen"] = str(val_HD[6][0])
        json_dict["meanHD"]["AVG"] = str((val_HD[0][0] + val_HD[0][1] + \
                                        val_HD[1][0] + val_HD[1][1] + \
                                        val_HD[2][0] + val_HD[2][1] + \
                                        val_HD[3][0] + val_HD[3][1] + \
                                        val_HD[4][1] + \
                                        val_HD[5][1] + \
                                        val_HD[6][0]) / 11.)

        json_dict["meanSD"]["liver"] = str(val_SD[0][0])
        json_dict["meanSD"]["liver_tumor"] = str(val_SD[0][1])
        json_dict["meanSD"]["kidney"] = str(val_SD[1][0])
        json_dict["meanSD"]["kidney_tumor"] = str(val_SD[1][1])
        json_dict["meanSD"]["hepatic_vessel"] = str(val_SD[2][0])
        json_dict["meanSD"]["hepatic_vessel_tumor"] = str(val_SD[2][1])
        json_dict["meanSD"]["pancreas"] = str(val_SD[3][0])
        json_dict["meanSD"]["pancreas_tumor"] = str(val_SD[3][1])
        json_dict["meanSD"]["colon"] = str(val_SD[4][1])
        json_dict["meanSD"]["lung"] = str(val_SD[5][1])
        json_dict["meanSD"]["spleen"] = str(val_SD[6][0])
        json_dict["meanSD"]["AVG"] = str((val_SD[0][0] + val_SD[0][1] + \
                                        val_SD[1][0] + val_SD[1][1] + \
                                        val_SD[2][0] + val_SD[2][1] + \
                                        val_SD[3][0] + val_SD[3][1] + \
                                        val_SD[4][1] + \
                                        val_SD[5][1] + \
                                        val_SD[6][0]) / 11.)

        print(json_dict["meanDice"])
        print(json_dict["meanHD"])
        print(json_dict["meanSD"])

        with open(os.path.join(args.save_path, "summary.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
