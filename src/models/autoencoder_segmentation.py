import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

################################
## Autoencoder Architecture ##
################################

class AutoEncoder(nn.Module):
    """
    Autoencoder architecture for image reconstruction.
    The encoder compresses the input image into a latent representation,
    and the decoder reconstructs the image from this representation.
    Args:
        init_features (int): Number of initial features for the encoder.
    """
    def __init__(self, init_features=128):
        super(AutoEncoder, self).__init__()
        
        f = init_features

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, f, kernel_size=3, stride=2, padding=1), # 3*512*512 > 64*256*256
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(f, f * 2, kernel_size=3, stride=2, padding=1), # 64*256*256 > 128*128*128
            nn.BatchNorm2d(f * 2),
            nn.ReLU(),
            nn.Conv2d(f * 2, f * 4, kernel_size=3, stride=2, padding=1), # 128*128*128 > 256*64*64
            nn.BatchNorm2d(f * 4),
            nn.ReLU(),
            nn.Conv2d(f * 4, f * 8, kernel_size=3, stride=2, padding=1), # 256*64*64 > 512*32*32
            nn.BatchNorm2d(f * 8),
            nn.ReLU(),
            nn.Conv2d(f * 8, f * 16, kernel_size=3, stride=2, padding=1), #512*32*32 > 1024*16*16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f * 16, f * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1024*16*16 > 512*32*32
            nn.BatchNorm2d(f * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(f * 8, f * 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 512*32*32 > 256*64*64
            nn.BatchNorm2d(f * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(f * 4, f * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256*64*64 > 128*128*128
            nn.BatchNorm2d(f * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(f * 2, f, kernel_size=3, stride=2, padding=1, output_padding=1),      # 128*128*128 > 64*256*256
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.ConvTranspose2d(f, 3, kernel_size=3, stride=2, padding=1, output_padding=1),          # 64*256*256 > 3*512*512
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

##############################
## Segmentation Decoder Head ##
##############################

class SegmentationDecoder(nn.Module):
    '''
    Segmentation Decoder for the Autoencoder.
    Args:
        input_channel (int): Number of input channels from the encoder.
        output_channel (int): Number of output channels for segmentation.
    '''
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channel, 512, 3, stride=2, padding=1, output_padding=1),  # 1024*16*16 > 512*32*32
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),   # 512*32*32 > 256*64*64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),   # 256*64*64 > 128*128*128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),    # 128*128*128 > 64*256*256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channel, 3, stride=2, padding=1, output_padding=1),  # 64*256*256 > 3*512*512
        )

    def forward(self, features):
        return self.seg_decoder(features)

###########################################
## Final Autoencoder-based Segmentation ##
###########################################

class AutoEncoderSegmentation(nn.Module):
    '''
    Autoencoder-based Segmentation Model.
    Args:
        encoder (nn.Module): Encoder part of the autoencoder.
        num_classes (int): Number of classes for segmentation.
    Returns:
        out (torch.Tensor): Segmentation output.
    '''
    def __init__(self, encoder, num_classes=3):
        super().__init__()
        # Frozen encoder from pre-trained autoencoder
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Segmentation Decoder
        self.seg_decoder = SegmentationDecoder(input_channel=self.encoder[-1].out_channels, output_channel=num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # Extract latent features
        out = self.seg_decoder(features)
        return out

########################################
## Autoencoder Pretraining Function  ##
########################################

def pretrain_autoencoder(autoencoder, train_loader, val_loader, 
                         save_dir, save_name,
                         num_epochs, device, patience):
    '''
    Pretrain the autoencoder on the training dataset.
    Args:
        autoencoder (nn.Module): Autoencoder model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        save_dir (str): Directory to save the pretrained model.
        save_name (str): Name of the saved model file.
        num_epochs (int): Number of epochs for pretraining.
        device (str): Device to use ("cuda" or "cpu").
        patience (int): Number of epochs with no improvement after which training will be stopped.
    Returns:
        history (dict): Dictionary containing training and validation losses.
    '''
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    save_path = Path(save_dir) / save_name

    # Initialize history dict to store losses
    history = {"train_loss": [], "val_loss": []}


    for epoch in range(num_epochs):
        # Training Phase
        autoencoder.train()
        running_loss = 0.0
        for images, _, _, _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
    
        # Validation Phase
        autoencoder.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, _, _, _ in tqdm(val_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs} [Val]",unit='image'):
                images = images.to(device)
                reconstructed = autoencoder(images)
                val_loss = criterion(reconstructed, images)
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)

        # Store the losses for plotting
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        # Print Losses
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping Check (on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = autoencoder.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Minimum Val MSE Loss: {best_val_loss:.6f}")
            break
    # save the model_state with the best performance
    torch.save(best_model_state, save_path)
    print(f"Best autoencoder model saved at {save_path}")

    return history
