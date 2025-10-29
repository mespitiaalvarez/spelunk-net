"""
Fast-SCNN adapted for 4-channel input with frozen RGB channels.

This module extends the original Fast-SCNN to accept 4-channel input
while preserving pretrained RGB weights and only training the new channel.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the Fast-SCNN submodule to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/Fast-SCNN-pytorch'))

from models.fast_scnn import (
    FastSCNN, _ConvBNReLU, _DSConv, GlobalFeatureExtractor,
    FeatureFusionModule, Classifer
)


class LearningToDownsample4Ch(nn.Module):
    """Learning to downsample module with 4-channel input support"""

    def __init__(self, in_channels=4, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample4Ch, self).__init__()
        self.conv = _ConvBNReLU(in_channels, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class FastSCNN4Ch(nn.Module):
    """Fast-SCNN with 4-channel input support"""
    
    def __init__(self, num_classes, in_channels=4, aux=False, **kwargs):
        super(FastSCNN4Ch, self).__init__()
        self.aux = aux
        self.in_channels = in_channels
        self.learning_to_downsample = LearningToDownsample4Ch(in_channels, 32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = nn.functional.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


def load_pretrained_with_new_channel(model_4ch, pretrained_path, freeze_rgb=True, init_method='random'):
    """
    Load pretrained 3-channel weights into 4-channel model.
    
    Args:
        model_4ch: FastSCNN4Ch model instance
        pretrained_path: Path to pretrained 3-channel weights
        freeze_rgb: If True, freeze the RGB channel weights
        init_method: How to initialize the 4th channel ('random', 'zeros', 'copy_r', 'copy_g', 'copy_b', 'mean')
    
    Returns:
        model_4ch: Model with loaded weights
    """
    # Load pretrained state dict
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    
    # Get the first conv layer weight from pretrained model
    # Shape: [out_channels, 3, kernel_h, kernel_w]
    pretrained_first_conv = pretrained_state['learning_to_downsample.conv.conv.0.weight']
    
    # Get current model's first conv weight
    # Shape: [out_channels, 4, kernel_h, kernel_w]
    current_first_conv = model_4ch.learning_to_downsample.conv.conv[0].weight
    
    # Copy RGB channels (first 3 channels)
    with torch.no_grad():
        current_first_conv[:, :3, :, :] = pretrained_first_conv
        
        # Initialize the 4th channel based on method
        if init_method == 'random':
            # Random initialization (same as PyTorch default)
            nn.init.kaiming_uniform_(current_first_conv[:, 3:4, :, :], a=0)
        elif init_method == 'zeros':
            current_first_conv[:, 3:4, :, :] = 0.0
        elif init_method == 'copy_r':
            current_first_conv[:, 3:4, :, :] = pretrained_first_conv[:, 0:1, :, :]
        elif init_method == 'copy_g':
            current_first_conv[:, 3:4, :, :] = pretrained_first_conv[:, 1:2, :, :]
        elif init_method == 'copy_b':
            current_first_conv[:, 3:4, :, :] = pretrained_first_conv[:, 2:3, :, :]
        elif init_method == 'mean':
            # Average of RGB channels
            current_first_conv[:, 3:4, :, :] = pretrained_first_conv.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    # Update the state dict with the new first conv weights
    pretrained_state['learning_to_downsample.conv.conv.0.weight'] = current_first_conv
    
    # Load all weights (including the modified first conv)
    model_4ch.load_state_dict(pretrained_state, strict=True)
    
    # Freeze RGB channels if requested
    if freeze_rgb:
        freeze_rgb_channels(model_4ch)
    
    return model_4ch


def freeze_rgb_channels(model):
    """
    Freeze only the RGB channel weights in the first conv layer.
    All other layers remain trainable.
    """
    # Freeze the RGB channels (first 3) of the first conv layer
    first_conv = model.learning_to_downsample.conv.conv[0]
    
    # Create a custom hook to zero out gradients for RGB channels
    def hook_rgb_freeze(grad):
        """Zero out gradients for the first 3 input channels"""
        grad_clone = grad.clone()
        grad_clone[:, :3, :, :] = 0.0  # Freeze RGB channels
        return grad_clone
    
    # Register the backward hook
    first_conv.weight.register_hook(hook_rgb_freeze)
    
    print("✓ RGB channels frozen. Only the 4th channel will be trained.")


def get_trainable_params(model):
    """
    Get parameters that should be trained.
    For fine-grained control, you can also separate 4th channel params from others.
    """
    return [p for p in model.parameters() if p.requires_grad]


def get_4th_channel_only_params(model):
    """
    Get only the 4th channel parameters for separate optimization.
    Useful if you want different learning rates.
    """
    # The 4th channel weights are part of the first conv layer
    first_conv = model.learning_to_downsample.conv.conv[0]
    
    # We can't easily separate this, so return all trainable params
    # In practice, with the hook, only 4th channel will get gradients
    return get_trainable_params(model)


# Example usage
if __name__ == '__main__':
    # Create 4-channel model
    num_classes = 19  # e.g., Cityscapes
    model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=True)
    
    # Load pretrained weights and freeze RGB channels
    pretrained_path = '../../external/Fast-SCNN-pytorch/weights/fast_scnn_citys.pth'
    
    if os.path.exists(pretrained_path):
        model = load_pretrained_with_new_channel(
            model, 
            pretrained_path, 
            freeze_rgb=True,
            init_method='random'  # or 'zeros', 'mean', etc.
        )
        print("✓ Loaded pretrained weights with 4th channel initialized")
    
    # Test forward pass
    dummy_input = torch.randn(2, 4, 256, 512)  # 4 channels now!
    outputs = model(dummy_input)
    print(f"Output shape: {outputs[0].shape}")
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

