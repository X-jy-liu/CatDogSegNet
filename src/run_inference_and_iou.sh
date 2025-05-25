#!/bin/bash
set -e

# Inference via autoencoder
python src/run_inference.py --image_dir Dataset/Val/color \
                            --mask_dir Dataset/Val/label \
                            --save_dir Dataset/Test/val_autoencoder_preds_unified_size \
                            --checkpoint_path params/train_autoencoder_segmentation_unified_size.pth \
                            --pretrain_path params/train_autoencoder_pretrain_unified_size.pth \
                            --device cuda \
                            --mode 1

# Inference via CLIP
python src/run_inference.py --image_dir Dataset/Val/color \
                            --mask_dir Dataset/Val/label \
                            --save_dir Dataset/Test/val_clip_preds_unified_size \
                            --checkpoint_path params/train_clip_segmentation_unified_size.pth \
                            --target_size 224 \
                            --device cuda \
                            --mode 2

# autoencoder
python src/calculate_IoU.py --gt_folder Dataset/Val/label \
                            --pred_folder Dataset/Test/val_autoencoder_preds_unified_size \
                            --output_file val_autoencoder_iou_unified_size.txt
# clip
python src/calculate_IoU.py --gt_folder Dataset/Val/label \
                            --pred_folder Dataset/Test/val_clip_preds_unified_size \
                            --output_file val_clip_iou_unified_size.txt