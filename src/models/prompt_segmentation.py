import torch 
import torch.nn as nn

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, init_features):
        super().__init__()
        f = init_features
        self.encoder1 = self._conv_block(in_channels, f)
        self.encoder2 = self._conv_block(f, f * 2)
        self.encoder3 = self._conv_block(f * 2, f * 4)
        self.encoder4 = self._conv_block(f * 4, f * 8)
        self.bottleneck = self._conv_block(f * 8, f * 16)
        self.bottleneck_channels = f * 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        return [enc1, enc2, enc3, enc4], bottleneck

class UNetDecoder(nn.Module):
    def __init__(self, bottleneck_channels, out_channels, init_features):
        super().__init__()
        f = init_features

        self.up4 = nn.ConvTranspose2d(bottleneck_channels, f * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(f * 8 + f * 8, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(f * 4 + f * 4, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(f * 2 + f * 2, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(f + f, f)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_feats, bottleneck):
        enc1, enc2, enc3, enc4 = enc_feats

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, enc4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, enc3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, enc2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, enc1], dim=1))

        return self.final(d1)


class PointPromptEncoder(nn.Module):
    """
    Encode prompt points into a spatial representation using pre-generated heatmaps
    """
    def __init__(self, input_channels=1, output_dim=1024, num_classes=3):
        """
        Args:
            input_channels (int): Number of input channels for the prompt heatmap.
            output_dim (int): Dimension of the final output representation.
            num_classes (int): Number of classes for point classification.
        """
        super().__init__()
        #  define the intermediate hidden dimension
        self.hidden_dim = output_dim // 2
        # Point class embedding (optional)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=output_dim)  # 3 classes (0=background, 1=cat, 2=dog)
        
        # Spatial encoding mechanism
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, output_dim, kernel_size=3, padding=1, stride=4)
        )
    
    def forward(self, prompt_heatmap, point_class=None):
        """
        Args:
            prompt_heatmap (torch.Tensor): Pre-generated heatmap from dataset (B, 1, H, W)
            point_class (torch.Tensor, optional): Class labels for the prompt points (B,)
        
        Returns:
            torch.Tensor: Encoded prompt embedding (B, output_dim, H, W)
        """
        B = prompt_heatmap.shape[0]
        
        # Spatially encode heatmap
        spatial_prompt = self.spatial_encoder(prompt_heatmap)
        
        # Incorporate class information if provided
        if point_class is not None:
            class_embed = self.class_embedding(point_class)  # (B, output_dim)
            class_embed = class_embed.view(B, -1, 1, 1)  # (B, output_dim, 1, 1)

            # Broadcast class embedding across spatial dimensions so the final output is globally class-aware
            class_embed = class_embed.expand(-1, -1, spatial_prompt.size(2), spatial_prompt.size(3))
            
            # Combine class information with spatial prompt
            return spatial_prompt + class_embed
        
        return spatial_prompt

class PromptSegmentation(nn.Module):
    def __init__(self, unet_in_channels=3, prompt_dim=1024, unet_init_features=64):
        super().__init__()
        
        # bottleneck features = init_features * 16 -> (64 * 16 = 1024)
        self.encoder = UNetEncoder(in_channels=unet_in_channels, init_features=unet_init_features)
        # prompt_encoder output channels = 64 * 16 = 1024
        self.prompt_encoder = PointPromptEncoder(input_channels=1, output_dim=prompt_dim, num_classes=3)  # 3 classes (0=background, 1=cat, 2=dog)
        # decoder expects (B, 1024 32, 32) from encoder concentated (B, 1024, 32, 32) from prompt_encoder
        self.decoder = UNetDecoder(bottleneck_channels=self.encoder.bottleneck_channels + prompt_dim,
                                   out_channels=1, init_features=unet_init_features)
        
    
    def forward(self, image, prompt_heatmap, point_class=None):
        # Step 1: Encode image -> get encoder features and bottleneck
        enc_features, image_bottleneck = self.encoder(image)

        # Step 2: Encode prompt heatmap
        prompt_features = self.prompt_encoder(prompt_heatmap, point_class)  # (B, prompt_dim, H/16, W/16)

        # Step 3: Concatenate prompt with image bottleneck
        combined_bottleneck = torch.cat([image_bottleneck, prompt_features], dim=1)

        # Step 4: Decode
        mask_logits = self.decoder(enc_features, combined_bottleneck)

        return mask_logits