#!/bin/bash
set -e
# # Run unet
# python src/main.py --mode 0 --epochs 200 --patience 5 | tee unet_segmentation.log
# Run autoencoder pretraining
# python src/main.py --img_dir "Dataset/TrainProcessed/color" \
#                    --msk_dir "Dataset/TrainProcessed/label" \
#                    --mode 1 --pretrain 1 --epochs 200 --patience 10 | tee autoencoder_pretrain_unified_size.log
# Run autoencoder segementation
python src/main.py --img_dir "Dataset/TrainProcessed/color" \
                   --msk_dir "Dataset/TrainProcessed/label" \
                   --mode 1 --pretrain 0 --epochs 30 --patience 50 | tee autoencoder_segmentation_unified_size.log

# Train clip_based segmentation model
python src/main.py --img_dir "Dataset/TrainProcessed/color" \
                   --msk_dir "Dataset/TrainProcessed/label" \
                   --mode 2 --epochs 30 --patience 50 | tee clip_seg_train_unified_size.log