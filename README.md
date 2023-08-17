# DoDNet
<p align="left">
    <img src="a_DynConv/dodnet.png" width="85%" height="85%">
</p>


This repo holds the pytorch implementation of DoDNet and TransDoDNet:<br />

**DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets** 
(https://arxiv.org/pdf/2011.10217.pdf) \
**Learning from partially labeled data for multi-organ and tumor segmentation** 
(https://arxiv.org/pdf/2211.06894.pdf)

<!-- ## Requirements
Python 3.7<br />
PyTorch==1.4.0<br />
<<<<<<< HEAD
[Apex==0.1](https://github.com/NVIDIA/apex)<br />
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)<br />
=======
Apex==0.1<br /> -->
>>>>>>> Add TransDoD

## Usage


<!--### 0. Installation
* Clone this repo
```
git clone https://github.com/jianpengz/DoDNet.git
```
-->

### 1. MOTS Dataset Preparation
Before starting, MOTS should be re-built from the serveral medical organ and tumor segmentation datasets

Partial-label task | Data source
--- | :---:
Liver | [data](https://competitions.codalab.org/competitions/17094)
Kidney | [data](https://kits19.grand-challenge.org/data/)
Hepatic Vessel | [data](http://medicaldecathlon.com/)
Pancreas | [data](http://medicaldecathlon.com/)
Colon | [data](http://medicaldecathlon.com/)
Lung | [data](http://medicaldecathlon.com/)
Spleen | [data](http://medicaldecathlon.com/)

<!-- * Download and put these datasets in `dataset/0123456/`. 
* Re-spacing the data by `python re_spacing.py`, the re-spaced data will be saved in `0123456_spacing_same/`.

The folder structure of dataset should be like

    dataset/0123456_spacing_same/
    ├── 0Liver
    |    └── imagesTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    |    └── labelsTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    ├── 1Kidney
    ├── ... -->
* Preprocessed data will be available soon.

### 2. Training/Testing/Evaluation
sh run_script.sh


<!-- ### 2. Model
Pretrained model is available in [checkpoint](https://drive.google.com/file/d/1qj8dJ_G1sHiCmJx_IQjACQhjUQnb4flg/view?usp=sharing)

### 3. Training
* cd `a_DynConv/' and run 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train.py \
--train_list='list/MOTS/MOTS_train.txt' \
--snapshot_dir='snapshots/dodnet' \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=2 \
--num_epochs=1000 \
--start_epoch=0 \
--learning_rate=1e-2 \
--num_classes=2 \
--num_workers=8 \
--weight_std=True \
--random_mirror=True \
--random_scale=True \
--FP16=False
```

### 4. Evaluation
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--val_list='list/MOTS/MOTS_test.txt' \
--reload_from_checkpoint=True \
--reload_path='snapshots/dodnet/MOTS_DynConv_checkpoint.pth' \
--save_path='outputs/' \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--num_workers=2
```

### 5. Post-processing
```
python postp.py --img_folder_path='outputs/dodnet/'
``` -->


### 3. Citation
If this code is helpful for your study, please cite:
```
@inproceedings{zhang2021dodnet,
  title={DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets},
  author={Zhang, Jianpeng and Xie, Yutong and Xia, Yong and Shen, Chunhua},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={},
  year={2021}
}
@article{xie2022learning,
  title={Learning from partially labeled data for multi-organ and tumor segmentation},
  author={Xie, Yutong and Zhang, Jianpeng and Xia, Yong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2211.06894},
  year={2022}
}
```