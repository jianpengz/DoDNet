import numpy as np
import os
import sys
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from joblib import Parallel, delayed

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

spacing = {
    0: [1.5, 0.8, 0.8],
    1: [1.5, 0.8, 0.8],
    2: [1.5, 0.8, 0.8],
    3: [1.5, 0.8, 0.8],
    4: [1.5, 0.8, 0.8],
    5: [1.5, 0.8, 0.8],
    6: [1.5, 0.8, 0.8],
}

base_path = '0123456_spacing_same'
ori_path = sys.argv[1] if len(sys.argv) > 1 else './0123456/'
new_path = sys.argv[2] if len(sys.argv) > 2 else f'./{base_path}'
n_procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4

count = -1


def process_msd(root3, i_files3):
    img_path = os.path.join(root3, i_files3)
    imageITK = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(imageITK)
    ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    task_id = int(i_dirs1[0])
    target_spacing = np.array(spacing[task_id])
    spc_ratio = ori_spacing / target_spacing

    data_type = image.dtype
    if i_dirs2 != 'labelsTr':
        data_type = np.int32

    if i_dirs2 == 'labelsTr':
        order = 0
        mode_ = 'edge'
    else:
        order = 3
        mode_ = 'constant'

    image = image.astype(float)

    image_resize = resize(image, (int(ori_size[0] * spc_ratio[0]), int(ori_size[1] * spc_ratio[1]),
                                  int(ori_size[2] * spc_ratio[2])),
                          order=order, mode=mode_, cval=0, clip=True, preserve_range=True)
    image_resize = np.round(image_resize).astype(data_type)
    image_resize = np.transpose(image_resize, (2, 1, 0))

    # save
    save_path = os.path.join(new_path, i_dirs1, i_dirs2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveITK = sitk.GetImageFromArray(image_resize)
    saveITK.SetSpacing(target_spacing)
    saveITK.SetOrigin(tuple(ori_origin[i] for i in [2, 1, 0]))
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(saveITK, os.path.join(save_path, i_files3))


def process_kidney(root3, i_dirs3):
    for root4, dirs4, files4 in os.walk(os.path.join(root3, i_dirs3)):
        for i_files4 in sorted(files4):
            img_path = os.path.join(root4, i_files4)
            imageITK = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(imageITK)
            ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
            ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
            ori_origin = imageITK.GetOrigin()
            ori_direction = imageITK.GetDirection()

            task_id = int(i_dirs1[0])
            target_spacing = np.array(spacing[task_id])

            if ori_spacing[0] < 0 or ori_spacing[1] < 0 or ori_spacing[2] < 0:
                print("error")
            spc_ratio = ori_spacing / target_spacing

            data_type = image.dtype
            if i_files4 != 'segmentation.nii.gz':
                data_type = np.int32

            if i_files4 == 'segmentation.nii.gz':
                order = 0
                mode_ = 'edge'
            else:
                order = 3
                mode_ = 'constant'

            image = image.astype(float)

            image_resize = resize(image, (
                int(ori_size[0] * spc_ratio[0]), int(ori_size[1] * spc_ratio[1]),
                int(ori_size[2] * spc_ratio[2])), order=order, cval=0, clip=True,
                                  preserve_range=True)

            image_resize = np.round(image_resize).astype(data_type)

            # save
            save_path = os.path.join(new_path, i_dirs1, 'origin', i_dirs3)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saveITK = sitk.GetImageFromArray(image_resize)
            saveITK.SetSpacing(target_spacing[[2, 1, 0]])
            saveITK.SetOrigin(ori_origin)
            saveITK.SetDirection(ori_direction)
            sitk.WriteImage(saveITK, os.path.join(save_path, i_files4))


#https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python/290494
def replace_path(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))

        copymode(file_path, abs_path)
        #Remove original file
        remove(file_path)
        #Move new file
        move(abs_path, file_path)

'''
for root1, dirs1, _ in os.walk(ori_path):
    print(dirs1)
    for i_dirs1 in tqdm(sorted(dirs1)):  # 0Liver
        print(i_dirs1)
        ###########################################################################
        if i_dirs1 == '1Kidney':
            for root2, dirs2, files2 in os.walk(os.path.join(root1, i_dirs1)):
                for root3, dirs3, files3 in os.walk(os.path.join(root2, 'origin')):
                    print(dirs3)
                    Parallel(n_jobs=n_procs, verbose=500)(delayed(process_kidney)(root3, i_dirs3) for i_dirs3 in sorted(dirs3) if i_dirs)
            continue
       #############################################################################
        for root2, dirs2, files2 in os.walk(os.path.join(root1, i_dirs1)):
            for i_dirs2 in sorted(dirs2):  # imagesTr
                for root3, dirs3, files3 in os.walk(os.path.join(root2, i_dirs2)):
                    Parallel(n_jobs=n_procs, verbose=50)(delayed(process_msd)(root3, i_files3) for i_files3 in sorted(files3) if i_files3[0] != '.')

    break
'''

if new_path != base_path:
    for id_file in ['list/MOTS/MOTS_train.txt', 'list/MOTS/MOTS_test.txt']:
        replace_path(id_file, base_path, new_path)
