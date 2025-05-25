#!/usr/bin/env python

import torch
import argparse
import clip
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data.preprocessing import *
from data.PetDataset import PetDataset, PetDatasetWithPrompt
# import unet related functions
from models.unet_segmentation import UNet
# import autoencoder related functions
from models.autoencoder_segmentation import (AutoEncoder, AutoEncoderSegmentation,
                                             pretrain_autoencoder)
# import CLIP related functions
from models.clip_segmentation import CLIPSegmentationModel
# import prompt based model
from models.prompt_segmentation import PromptSegmentation
from utils.loss_functions import CombinedFocalDiceLoss, BinaryFocalLoss, IouLoss
from utils.training_plot import training_plot
from utils.training import training, prompt_training

# Define Arguments
parser = argparse.ArgumentParser(description="Train segmentation model")
parser.add_argument("--img_dir", type=str, default="Dataset/Processed/color", help="Directory for training images")
parser.add_argument("--msk_dir", type=str, default="Dataset/Processed/label", help="Directory for training masks")
parser.add_argument("--pnt_dir", type=str, default="Dataset/ProcessedWithPrompt/color/points", help="Directory for training prompt points")
parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], required=True, help="0 for AutoEncoder, 1 for CLIP-based segmentation")
parser.add_argument("--pretrain", type=int, choices=[0, 1], default=0, help="1 to pretrain autoencoder")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--save_dir", type=str, default="params/", help="Path to save the model")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args = parser.parse_args()

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training run on: "{device}"')
print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('no cuda devices')
# Ensure save directory exists
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# Load Dataset Paths
img_paths = sorted(Path(args.img_dir).glob("*.*"))
msk_paths = sorted(Path(args.msk_dir).glob("*.*"))
pnt_paths = sorted(Path(args.pnt_dir).glob("*.*"))

if args.mode == 3:
    train_img_paths, val_img_paths, \
    train_msk_paths, val_msk_paths, \
    train_pnt_paths, val_pnt_paths  = train_test_split(img_paths, msk_paths, pnt_paths, test_size=0.2, random_state=42)
else:
    train_img_paths, val_img_paths, \
    train_msk_paths, val_msk_paths  = train_test_split(img_paths, msk_paths, test_size=0.2, random_state=42)

# Select Model

###############################
############ U-Net ############
###############################
if args.mode == 0:
    print("Training Unet-Based Segmentation Model ...")
    # define the model
    model = UNet()

    # define training data split
    train_dataset = PetDataset(
                img_paths=train_img_paths,
                msk_paths=train_msk_paths,
                transform=standard_transform)
    val_dataset = PetDataset(
                    img_paths=val_img_paths,
                    msk_paths=val_msk_paths,
                    transform=standard_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # define save paths
    train_plot = "unet_segmentation.png"
    training_plot_save_name = Path(args.save_dir) / train_plot
    save_name = "unet_segmentation.pth"

    # Define Loss Functions and Optimizer
    criterion= CombinedFocalDiceLoss(alpha=0.5, gamma=2.0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train Model
    history = training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=criterion,
        val_criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        save_name=save_name,
        patience=args.patience
    )
    
    training_plot(history,save_path=training_plot_save_name)

###############################
######### Autoencoder #########
###############################
elif args.mode == 1:
    # define training data split
    train_dataset = PetDataset(
                img_paths=train_img_paths,
                msk_paths=train_msk_paths,
                transform=standard_transform)
    val_dataset = PetDataset(
                    img_paths=val_img_paths,
                    msk_paths=val_msk_paths,
                    transform=standard_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # define config file name
    pretrain_model_save_name = "train_autoencoder_pretrain_unified_size.pth"
    model_save_name = "train_autoencoder_segmentation_unified_size.pth"
    pretrain_plot = "train_autoencoder_pretrain_unified_size.png"
    seg_train_plot = "train_autoencoder_segmentation_unified_size.png"
    if args.pretrain == 1: # the pretrain mode
        print('Autoencoder Pretrain ...')
        autoencoder = AutoEncoder()
        training_plot_save_name = Path(args.save_dir) / pretrain_plot
        history = pretrain_autoencoder(autoencoder, train_loader, val_loader, num_epochs=args.epochs, 
                                       save_dir=args.save_dir, save_name=pretrain_model_save_name,
                                       device=device, patience=args.patience)
        training_plot(history,save_path=training_plot_save_name)
        print("Autoencoder pretrained and saved.")
    else: # the segmentation head train mode (encode freezed)
        print('Training Autoencoder-Based Segmentation Model ...')
        autoencoder = AutoEncoder()
        pretrain_config_path = Path(args.save_dir) / pretrain_model_save_name
        pretrain_state_dict = torch.load(pretrain_config_path, map_location=device) 
        autoencoder.load_state_dict(pretrain_state_dict)
        training_plot_save_name = Path(args.save_dir) / seg_train_plot
        model = AutoEncoderSegmentation(autoencoder.encoder, num_classes=3)
        train_criterion = CombinedFocalDiceLoss()
        val_criterion = IouLoss()
        # filter(function,iterable) and lambda arguments: expression freeze the pretraining
        #  and train the decoder only
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        history = training(
            model, train_loader, val_loader,
            train_criterion=train_criterion, val_criterion=val_criterion,
            optimizer=optimizer,
            num_epochs=args.epochs, device=device,
            save_dir=args.save_dir,
            save_name=model_save_name,
            patience=args.patience
        )
        training_plot(history,save_path=training_plot_save_name)
        print("Autoencoder pretrained and saved.")

###############################
############ ClIP #############
###############################
elif args.mode == 2:
    print("Training CLIP-Based Segmentation Model ...")
    clip_model, _ = clip.load("RN50", device=device)
    model = CLIPSegmentationModel(clip_model, num_classes=3)
    train_dataset = PetDataset(
                img_paths=train_img_paths,
                msk_paths=train_msk_paths,
                resize_fn=resize_with_padding,
                resize_target_size=224,
                transform=clip_transform)
    val_dataset = PetDataset(
                    img_paths=val_img_paths,
                    msk_paths=val_msk_paths,
                    resize_fn=resize_with_padding,
                    resize_target_size=224,
                    transform=clip_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # define save paths
    train_plot = "train_clip_segmentation_unified_size.png"
    training_plot_save_name = Path(args.save_dir) / train_plot
    save_name = "train_clip_segmentation_unified_size.pth"

    # Define Loss Functions and Optimizer
    train_criterion = CombinedFocalDiceLoss()
    val_criterion = IouLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train Model
    history = training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=train_criterion,
        val_criterion=val_criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        save_name=save_name,
        patience=args.patience
    )
    
    training_plot(history,save_path=training_plot_save_name)

elif args.mode == 3:
    print('Training Prompt-based Segmentation model ...')
    # define model
    model = PromptSegmentation(unet_in_channels=3,
                               prompt_dim=1024,
                               unet_init_features=64)
    train_dataset = PetDatasetWithPrompt(
                img_paths=train_img_paths,
                msk_paths=train_msk_paths,
                pnt_paths= train_pnt_paths,
                transform=standard_transform)
    val_dataset = PetDatasetWithPrompt(
                    img_paths=val_img_paths,
                    msk_paths=val_msk_paths,
                    pnt_paths = val_pnt_paths,
                    transform=standard_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # define save paths
    train_plot = "prompt_unet_segmentation.png"
    training_plot_save_name = Path(args.save_dir) / train_plot
    save_name = "prompt_unet_segmentation.pth"

    # Define Loss Functions and Optimizer
    criterion= BinaryFocalLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train Model
    history = prompt_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=criterion,
        val_criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        save_name=save_name,
        patience=args.patience
    )
    
    training_plot(history,save_path=training_plot_save_name)

else:
    raise ValueError("Invalid mode. Use 0 for U-Net, 1 for autoencoder-based segmentation, 2 for CLIP-based segmentation or 3 for prompt segmentation.")