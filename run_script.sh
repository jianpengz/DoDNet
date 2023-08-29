cd TransDoD/

# Training
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train.py \
--train_list='MOTS/MOTS_train.txt' \
--snapshot_dir='snapshots/TransDoDNet/' \
--nnUNet_preprocessed='/media/userdisk2/jpzhang/nnU_data_SSD/nnUNet_preprocessed' \
--input_size='64,192,192' \
--learning_rate=2e-4 \
--batch_size=2 \
--num_gpus=2 \
--num_epochs=1000

# Testing
cd TransDoD/
CUDA_VISIBLE_DEVICES=6 python test.py \
--val_list='MOTS/MOTS_test.txt' \
--nnUNet_preprocessed='/media/userdisk2/jpzhang/nnU_data_SSD/nnUNet_preprocessed' \
--reload_path='/home/jpzhang/faster1/jpzhang/myproject-Seg/MOTS-pro2/f_transformers/snapshots_a/TrDoD_e3d3_level3_mem2_dynhead_3_8_c192_seed123_v1/checkpoint.pth' \
--reload_from_checkpoint=True \
--save_path='outputs/TransDoDNet' \
--input_size='64,192,192' 