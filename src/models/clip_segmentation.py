# The related blocks of CLIP segmentation models are store all-in-one here

import torch
import torch.nn as nn
######################
# Feature Extraction #
######################

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.conv1 = visual.conv1
        self.bn1 = visual.bn1
        self.relu1 = visual.relu1
        self.conv2 = visual.conv2
        self.bn2 = visual.bn2
        self.relu2 = visual.relu2
        self.conv3 = visual.conv3
        self.bn3 = visual.bn3
        self.relu3 = visual.relu3

        self.layer1 = visual.layer1
        self.layer2 = visual.layer2
        self.layer3 = visual.layer3
        self.layer4 = visual.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

#######################
## Segmentation Head ##
#######################

class CLIPSegmentationHead(nn.Module):
    """
    Updated segmentation head for ModifiedResNet in CLIP.
    Adjusted for 2048 input channels and 14x14 input spatial size.
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # Reduce channels first
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(input_channels, 1024, kernel_size=1),  # Reduce channels 2048 > 1024
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 > 28x28
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # 112x112 > 224x224
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Final segmentation layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)  # Output segmentation map

    def forward(self, x):
        x = self.reduce_channels(x)  # Reduce channels: (B, 20248, 14, 14) > (B, 1024, 14, 14)
        x1 = self.upsample1(x)  # 14x14 > 28x28
        x2 = self.upsample2(x1) # 112x112 > 224x224
        out = self.final_conv(x2)  

        return out
    
#####################################
### Final CLIP Segmentation model ###
#####################################

class CLIPSegmentationModel(nn.Module):
    """
    Lightweight CLIP-based segmentation model using only the necessary vision layers from RN50.
    """
    def __init__(self, clip_model, num_classes):
        super().__init__()

        # Replace full CLIP model with a custom extractor that only includes needed layers
        self.feature_extractor = CLIPFeatureExtractor(clip_model)  # defined above
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze backbone

        self.seg_head = CLIPSegmentationHead(input_channels=2048, output_channels=num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.seg_head(x)
        return x
