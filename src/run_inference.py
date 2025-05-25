#!/usr/bin/env python

import argparse
import torch
import os
import clip
from pathlib import Path

# import models
# Import your models
from models.unet_segmentation import UNet
from models.autoencoder_segmentation import AutoEncoder, AutoEncoderSegmentation  # <-- Import your Autoencoder models
from models.clip_segmentation import CLIPSegmentationModel
from models.prompt_segmentation import PromptSegmentation
from utils.inference import inference, promptInference  # Ensure you have a separate inference function

# Define function to load the correct model
def load_model(mode, checkpoint_path, device, pretrain_path=None):
    """
    Load the segmentation model based on the selected mode.

    Args:
        mode (int): 0 for U-Net, 1 for Autoencoder, 2 for CLIP.
        checkpoint_path (str): Path to the trained model checkpoint.
        device (str): Device to use ("cuda" or "cpu").

    Returns:
        model (torch.nn.Module): The loaded model.
    """

    #################################
    ############ U-Net ##############
    #################################
    if mode == 0:
        print("Loading U-Net model... ")
        model = UNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    
    #################################
    ########## Autoencoder ##########
    #################################
    elif mode == 1:
        print("Loading Autoencoder Segmentation Model...")

        if pretrain_path is None:
            raise ValueError("⚠️ Pretrained autoencoder path must be provided in Autoencoder mode (--pretrain_path).")

        # Load pretrained autoencoder
        autoencoder = AutoEncoder()
        pretrain_checkpoint = torch.load(pretrain_path, map_location=device)
        autoencoder.load_state_dict(pretrain_checkpoint)
        autoencoder.to(device)

        # Wrap the frozen encoder with the segmentation head
        model = AutoEncoderSegmentation(encoder=autoencoder.encoder, num_classes=3)

        # Load segmentation checkpoint (contains trained decoder)
        seg_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(seg_checkpoint)

        model.to(device)
        model.eval()
        return model

    #################################
    ############ CLIP ###############
    #################################
    elif mode == 2:
        print("Loading CLIP Segmentation model...")
        clip_model, _ = clip.load("RN50", device=device)
        model = CLIPSegmentationModel(clip_model=clip_model, num_classes=3)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.float()
        model.eval()
        return model

    ################
    # Prompt-based #
    ################
    elif mode ==3:
        print("Loading Prompt Segementation model...")
        model = PromptSegmentation()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.float()
        model.eval()
        return model
    
    else:
        raise ValueError("Invalid mode! Use 0 for U-Net, 1 for Autoencoder, 2 for CLIP or 3 for prompt model.")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for image segmentation.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--point_dir", type=str, default=None, help="Path to prompt points (only for prompt model)")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save output masks")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to pretrained autoencoder (required for autoencoder mode)")
    parser.add_argument("--target_size", type=int, default=512, help="Target input size for model")
    parser.add_argument("--device", type=str, required=True, help="Device for inference (cuda/cpu)")
    parser.add_argument("--mode", type=int, required=True, help="Model selection: 0 for U-Net, 1 for Autoencoder, 2 for CLIP, 3 for prompt model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = load_model(args.mode, args.checkpoint_path, args.device, pretrain_path=args.pretrain_path)
    # Run inference
    if args.mode in [0,1,2]:
        inference(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            save_dir=args.save_dir,
            mode = args.mode,
            input_image_size=args.target_size,
            model=model,
            device=args.device
        )
    elif args.mode == 3:
        promptInference(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            point_dir= args.point_dir,
            gt_save_dir= Path(args.save_dir) / "gt/",
            pred_save_dir=Path(args.save_dir) / "pred/",
            threshold=0.5,
            input_image_size=args.target_size,
            model=model,
            device=args.device
        )
    else:
        raise ValueError("Invalid mode! Use 0 for U-Net, 1 for Autoencoder, 2 for CLIP or 3 for prompt model.")