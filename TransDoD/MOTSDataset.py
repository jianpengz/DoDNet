import os
import os.path as osp
import numpy as np
import random
from torch.utils import data
import math
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
import pickle

class MOTSTrainDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        print("Start preprocessing....")
        if not max_iters == None:
            self.img_ids = self.img_ids * int(max_iters)
        
        self.files = []

        for item in self.img_ids:
            # print(item)
            img_identifier = item[0]

            if img_identifier[:4] == 'live':
                task_id = 0
            elif img_identifier[:4] == 'kidn':
                task_id = 1
            elif img_identifier[:4] == 'hepa':
                task_id = 2
            elif img_identifier[:4] == 'panc':
                task_id = 3
            elif img_identifier[:4] == 'colo':
                task_id = 4
            elif img_identifier[:4] == 'lung':
                task_id = 5
            elif img_identifier[:4] == 'sple':
                task_id = 6
            else:
                print("!!!Cannot find the task in MOTSDataset.py!!!")
            preprocessed_data_path = os.environ['nnUNet_preprocessed']
            img_gt_file = osp.join(preprocessed_data_path, 'Task100_MOTS/nnUNetData_plans_v2.1_stage1_FP16', img_identifier + '.npz')
            with open(os.path.join(preprocessed_data_path, 'Task100_MOTS/nnUNetData_plans_v2.1_stage1_FP16', img_identifier + '.pkl'), 'rb') as f:
                properties = pickle.load(f)
            for key in list(properties["class_locations"].keys()):
                if len(properties["class_locations"][key])==0:
                    del properties["class_locations"][key]
            self.files.append({
                "img_lab": img_gt_file,
                "name": img_identifier,
                "task_id": task_id,
                "properties": properties
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, class_locs):

        img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(len(class_locs)) + 1
            if len(class_locs[selected_class]) == 0:
                # if no foreground found, then randomly select
                d0 = random.randint(0, img_d - self.crop_d)
                h0 = random.randint(0, img_h - self.crop_h)
                w0 = random.randint(0, img_w - self.crop_w)
                d1 = d0 + self.crop_d
                h1 = h0 + self.crop_h
                w1 = w0 + self.crop_w
            else:
                selected_voxel = class_locs[selected_class][np.random.choice(len(class_locs[selected_class]))]
                center_d, center_h, center_w = selected_voxel

                d0 = center_d - self.crop_d // 2
                d1 = center_d + self.crop_d // 2
                h0 = center_h - self.crop_h // 2
                h1 = center_h + self.crop_h // 2
                w0 = center_w - self.crop_w // 2
                w1 = center_w + self.crop_w // 2

                if h0 < 0:
                    delta = h0 - 0
                    h0 = 0
                    h1 = h1 - delta
                if h1 > img_h:
                    delta = h1 - img_h
                    h0 = h0 - delta
                    h1 = img_h
                if w0 < 0:
                    delta = w0 - 0
                    w0 = 0
                    w1 = w1 - delta
                if w1 > img_w:
                    delta = w1 - img_w
                    w0 = w0 - delta
                    w1 = img_w
                if d0 < 0:
                    delta = d0 - 0
                    d0 = 0
                    d1 = d1 - delta
                if d1 > img_d:
                    delta = d1 - img_d
                    d0 = d0 - delta
                    d1 = img_d

        else:
            d0 = random.randint(0, img_d - self.crop_d)
            h0 = random.randint(0, img_h - self.crop_h)
            w0 = random.randint(0, img_w - self.crop_w)
            d1 = d0 + self.crop_d
            h1 = h0 + self.crop_h
            w1 = w0 + self.crop_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        img_lab = np.load(datafiles["img_lab"])['data']
        image = img_lab[0]
        label = img_lab[1]
        name = datafiles["name"]
        task_id = datafiles["task_id"]
        class_locs = datafiles["properties"]["class_locations"]

        image = self.pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
        label = self.pad_image(label, [self.crop_d, self.crop_h, self.crop_w])

        [d0, d1, h0, h1, w0, w1] = self.locate_bbx(label, class_locs)

        image = image[d0: d1, h0: h1, w0: w1]
        label = label[d0: d1, h0: h1, w0: w1]

        label = self.id2trainId(label, task_id)

        image = image[np.newaxis, :]

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, task_id


class MOTSTestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for item in self.img_ids:
            # print(item)
            img_identifier = item[0]

            if img_identifier[:4] == 'live':
                task_id = 0
            elif img_identifier[:4] == 'kidn':
                task_id = 1
            elif img_identifier[:4] == 'hepa':
                task_id = 2
            elif img_identifier[:4] == 'panc':
                task_id = 3
            elif img_identifier[:4] == 'colo':
                task_id = 4
            elif img_identifier[:4] == 'lung':
                task_id = 5
            elif img_identifier[:4] == 'sple':
                task_id = 6
            else:
                print("!!!Cannot find the task in MOTSDataset.py!!!")
            preprocessed_data_path = os.environ['nnUNet_preprocessed']
            img_gt_file = osp.join(preprocessed_data_path, 'Task100_MOTS/nnUNetData_plans_v2.1_stage1_FP16', img_identifier + '.npz')
            with open(os.path.join(preprocessed_data_path, 'Task100_MOTS/nnUNetData_plans_v2.1_stage1_FP16', img_identifier + '.pkl'), 'rb') as f:
                properties = pickle.load(f)
            for key in list(properties["class_locations"].keys()):
                if len(properties["class_locations"][key])==0:
                    del properties["class_locations"][key]
            self.files.append({
                "img_lab": img_gt_file,
                "name": img_identifier,
                "task_id": task_id,
                "properties": properties
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)


    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        img_lab = np.load(datafiles["img_lab"])['data']
        image = img_lab[0]
        label = img_lab[1]
        name = datafiles["name"]
        task_id = datafiles["task_id"]
        properties = datafiles["properties"]

        image = self.pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
        label = self.pad_image(label, [self.crop_d, self.crop_h, self.crop_w])

        image = image[np.newaxis, :]

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, task_id, properties


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            # do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=False,
            # angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            # angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            # angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=0,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.0, p_scale_per_sample=0.2, p_rot_per_sample=0.0,
            independent_scale_for_each_axis=False,
            data_key="image", label_key="label")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label"))

    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label, name, task_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)
    data_dict = {'image': image, 'label': label, 'name': name, 'task_id': task_id}
    tr_transforms = get_train_transform(patch_size=label.shape[2:])
    data_dict = tr_transforms(**data_dict)

    for i in range(data_dict['label'].shape[0]):
        task_id_i = data_dict['task_id'][i]
        if (task_id_i == 4) or (task_id_i == 5):
            data_dict['label'][i, 0] = -1
        if task_id_i == 6:
            data_dict['label'][i, 1] = -1

    return data_dict
