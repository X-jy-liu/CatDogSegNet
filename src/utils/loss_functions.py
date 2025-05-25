import torch.nn as nn
import torch
import torch.nn.functional as F
######################
### Loss Functions ###
######################

class DiceLoss(nn.Module):
    '''
    calculate the (1 - Dice) loss
    Args:
        ouputs(torch.Tensor): model outputs, shape (B, 3, 512, 512)
        masks(torch.Tensor): ground truth, shape (B, 512, 512)
    Returns:
        (1 - mean_dice_over_classes) Loss
    '''
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        # Convert logits to probabilities using softmax
        outputs = F.softmax(outputs, dim=1)

        # Convert masks to one-hot encoding
        masks_one_hot = F.one_hot(masks.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        # Compute Dice Score
        intersection = torch.sum(outputs * masks_one_hot, dim=(2, 3))
        union = torch.sum(outputs, dim=(2, 3)) + torch.sum(masks_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

class IouLoss(nn.Module):
    '''
    calculate the (1 - IoU) loss
    Args:
        ouputs(torch.Tensor): model outputs, shape (B, 3, 512, 512)
        masks(torch.Tensor): ground truth, shape (B, 512, 512)
    Returns:
        (1 - mean_iou_over_classes) Loss
    '''
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, outputs, masks):
        # Convert logits to probabilities using softmax
        outputs = F.softmax(outputs, dim=1)

        # Convert masks to one-hot encoding
        masks_one_hot = F.one_hot(masks.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        # Compute IoU Score
        intersection = torch.sum(outputs * masks_one_hot, dim=(2, 3))
        union = torch.sum(outputs + masks_one_hot, dim=(2, 3)) - intersection

        iou_score = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou_score.mean()

    def forward(self, outputs, masks):
        # Convert logits to probabilities using softmax
        outputs = F.softmax(outputs, dim=1)

        # Convert masks to one-hot encoding
        masks_one_hot = F.one_hot(masks.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        # Compute Dice Score
        intersection = torch.sum(outputs * masks_one_hot, dim=(2, 3))
        union = torch.sum(outputs, dim=(2, 3)) + torch.sum(masks_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Args:
        gamma (float): Focusing parameter (>1 focuses more on hard samples).
        alpha (tensor, optional): Class weighting (for additional balance).
        reduction (str): 'mean' or 'sum'.
    Returns:
        torch.Tensor: Focal loss.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter (>1 focuses more on hard samples).
            alpha (tensor, optional): Class weighting (for additional balance).
            reduction (str): 'mean' or 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model logits (B, C, H, W).
            targets (torch.Tensor): Ground truth (B, H, W).

        Returns:
            torch.Tensor: Focal loss.
        """
        log_probs = F.log_softmax(inputs, dim=1)  # Convert logits to log-probs
        probs = torch.exp(log_probs)              # Convert to probabilities

        # Gather log-probabilities of correct class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1))  # (B, 1, H, W)
        target_probs = probs.gather(1, targets.unsqueeze(1))          

        # Compute Focal Loss term
        focal_weight = (1 - target_probs) ** self.gamma  # Focus on hard examples

        if self.alpha is not None:
            alpha_factor = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal_weight = focal_weight * alpha_factor

        loss = -focal_weight * target_log_probs  # Apply weighting

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedFocalDiceLoss(nn.Module):
    """
    Combined Loss = (1 - alpha)*FocalLoss + alpha*DiceLoss
    Args:
            alpha (float): Balance between Focal and Dice Loss.
            gamma (float): Focal Loss focusing parameter.
            focal_alpha (tensor, optional): Class weighting for Focal Loss.
            smooth (float): Smoothing factor for Dice Loss.
    Returns:
            torch.Tensor: Combined loss.
    """
    def __init__(self, alpha=0.5, gamma=2.0, focal_alpha=None, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma=gamma, alpha=focal_alpha)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, outputs, masks):
        fl = self.focal_loss(outputs, masks) 
        dl = self.dice_loss(outputs, masks)
        return (1 - self.alpha) * fl + self.alpha * dl

# Binary Loss Function
class BinaryFocalLoss(nn.Module):
    '''
    Focal Loss for binary classification in prompt-based training.
    Args:
        alpha (float): Balance between positive and negative classes.
        gamma (float): Focusing parameter (>1 focuses more on hard samples).
        reduction (str): 'mean' or 'sum'.
    Returns:
        torch.Tensor: Focal loss.
    '''
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)  # avoid log(0)

        # Focal loss formula
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss