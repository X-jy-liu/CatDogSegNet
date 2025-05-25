import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()
        
        features = init_features
        # Encoder blocks
        self.encoder1 = self._conv_block(in_channels, features)
        self.encoder2 = self._conv_block(features, features * 2)
        self.encoder3 = self._conv_block(features * 2, features * 4)
        self.encoder4 = self._conv_block(features * 4, features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features * 8, features * 16)
        
        # Decoder upsampling layers (now ConvTranspose2d)
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(features * 16, features * 8)
        
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(features * 4, features * 2)
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(features * 2, features)
        
        # Final output layer
        self.final_layer = nn.Conv2d(features, out_channels, kernel_size=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # (B, features, 512, 512)
        enc2 = self.encoder2(self.pool(enc1))  # (B, features*2, 256, 256)
        enc3 = self.encoder3(self.pool(enc2))  # (B, features*4, 128, 128)
        enc4 = self.encoder4(self.pool(enc3))  # (B, features*8, 64, 64)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # (B, features*16, 32, 32)
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)  # (B, features*8, 64, 64)
        dec4 = torch.cat((dec4, enc4), dim=1)  # (B, features*16, 64, 64)
        dec4 = self.decoder4(dec4)

        dec3 = self.up3(dec4)  # (B, features*4, 128, 128)
        dec3 = torch.cat((dec3, enc3), dim=1)  # (B, features*8, 128, 128)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)  # (B, features*2, 256, 256)
        dec2 = torch.cat((dec2, enc2), dim=1)  # (B, features*4, 256, 256)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)  # (B, features, 512, 512)
        dec1 = torch.cat((dec1, enc1), dim=1)  # (B, features*2, 512, 512)
        dec1 = self.decoder1(dec1)

        # Final output layer
        return self.final_layer(dec1)
