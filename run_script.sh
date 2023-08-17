# Training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM TransDoD/train.py \
--train_list='MOTS/MOTS_train.txt' \
--snapshot_dir='snapshots/TransDoDNet/' \
--nnUNet_preprocessed='Path to nnUNet_preprocessed' \
--input_size='64,192,192' \
--learning_rate=2e-4 \
--batch_size=2 \
--num_gpus=2 \
--num_epochs=1000

# Testing
CUDA_VISIBLE_DEVICES=0 python TransDoD/test.py \
--val_list='MOTS/MOTS_test.txt' \
--nnUNet_preprocessed='Path to nnUNet_preprocessed' \
--reload_path='snapshots/TransDoDNet/checkpoint.pth' \
--reload_from_checkpoint=True \
--save_path='outputs/TransDoDNet' \
--input_size='64,192,192' 