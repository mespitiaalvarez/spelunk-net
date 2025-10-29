"""
PyTorch Dataset for cone and ball segmentation with 4-channel input (RGB + Depth).
Designed to work with the perception sim generated training data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ConesBallsDataset(Dataset):
    """
    PyTorch Dataset for cone and ball segmentation with depth.
    
    Returns:
        rgbd: (4, H, W) tensor - RGB + Depth combined, normalized
        mask: (H, W) tensor - integer labels 0-10
    
    The 4 channels are:
        - Channel 0: Red (normalized to [0, 1])
        - Channel 1: Green (normalized to [0, 1])
        - Channel 2: Blue (normalized to [0, 1])
        - Channel 3: Depth (in meters, can be scaled)
    """
    
    def __init__(self, data_dir, transform=None, depth_scale=1.0, max_depth=5.0):
        """
        Args:
            data_dir: Root directory containing rgb/, depth/, masks/ subdirectories
            transform: Optional transform to apply to RGBD input
            depth_scale: Scale factor for depth values (default: 1.0 keeps in meters)
            max_depth: Maximum depth value for normalization (default: 5.0 meters)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        
        # Find all samples
        self.samples = []
        rgb_dir = os.path.join(data_dir, "rgb")
        if not os.path.exists(rgb_dir):
            raise ValueError(f"RGB directory not found: {rgb_dir}")
        
        for filename in sorted(os.listdir(rgb_dir)):
            if filename.endswith(".png"):
                sample_id = filename[:-4]  # Remove .png
                self.samples.append(sample_id)
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {rgb_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load RGB
        rgb_path = os.path.join(self.data_dir, "rgb", f"{sample_id}.png")
        rgb = np.array(Image.open(rgb_path))
        
        # Handle RGBA if present
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W), [0, 1]
        
        # Load depth (16-bit PNG in millimeters)
        depth_path = os.path.join(self.data_dir, "depth", f"{sample_id}.png")
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # Convert to meters
        depth = torch.from_numpy(depth).unsqueeze(0) * self.depth_scale  # (1, H, W)
        
        # Load mask
        mask_path = os.path.join(self.data_dir, "masks", f"{sample_id}.npy")
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        # Combine RGB + Depth into 4-channel input
        rgbd = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        
        # Apply transforms if specified
        if self.transform:
            # Note: transforms should handle 4-channel input
            rgbd = self.transform(rgbd)
        
        return rgbd, mask


class ConesBallsDatasetSeparate(Dataset):
    """
    Alternative dataset that returns RGB and Depth separately.
    Use this if you want to process them differently before concatenation.
    
    Returns:
        rgb: (3, H, W) tensor
        depth: (1, H, W) tensor
        mask: (H, W) tensor
    """
    
    def __init__(self, data_dir, transform=None, depth_scale=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.depth_scale = depth_scale
        
        # Find all samples
        self.samples = []
        rgb_dir = os.path.join(data_dir, "rgb")
        for filename in sorted(os.listdir(rgb_dir)):
            if filename.endswith(".png"):
                sample_id = filename[:-4]
                self.samples.append(sample_id)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load RGB
        rgb_path = os.path.join(self.data_dir, "rgb", f"{sample_id}.png")
        rgb = np.array(Image.open(rgb_path))
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # Load depth
        depth_path = os.path.join(self.data_dir, "depth", f"{sample_id}.png")
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth).unsqueeze(0) * self.depth_scale
        
        # Load mask
        mask_path = os.path.join(self.data_dir, "masks", f"{sample_id}.npy")
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        
        return rgb, depth, mask


if __name__ == '__main__':
    # Test the dataset
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../../src/perception_sim/training_images"
    
    print(f"Testing dataset with data_dir: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)
    
    # Test 4-channel dataset
    dataset = ConesBallsDataset(data_dir)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Load a sample
    rgbd, mask = dataset[0]
    print(f"\nSample 0:")
    print(f"  RGBD shape: {rgbd.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  RGB channels range: [{rgbd[:3].min():.3f}, {rgbd[:3].max():.3f}]")
    print(f"  Depth channel range: [{rgbd[3].min():.3f}, {rgbd[3].max():.3f}] meters")
    print(f"  Unique labels: {torch.unique(mask).tolist()}")
    
    # Test batch loading
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    rgbd_batch, mask_batch = next(iter(dataloader))
    print(f"\nBatch test:")
    print(f"  RGBD batch shape: {rgbd_batch.shape}")
    print(f"  Mask batch shape: {mask_batch.shape}")
    print(f"✓ Dataset working correctly!")

