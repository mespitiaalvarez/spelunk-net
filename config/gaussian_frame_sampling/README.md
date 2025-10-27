# Frame Sampling Configuration Files

This directory contains JSON configuration files for specifying timestamps and parameters for sampling frames from MCAP bags.

## Two Sampling Methods

### 1. **Gaussian Sampling** (`mcap_to_imgs_gaussian.py`)
- Samples frames with higher probability around target timestamps
- May include temporally redundant frames (frames very close in time)
- Good for quick exploration

### 2. **Hard Sampling** (`mcap_to_imgs_hard_sampling.py`) ⭐ **RECOMMENDED**
- Gaussian sampling + minimum time gap enforcement
- Eliminates temporal redundancy (avoids nearly-identical consecutive frames)
- More efficient for training - better diversity, less overfitting
- **Use this if you haven't started labeling/training yet!**

## Configuration Format

```json
{
  "description": "Human-readable description of what these timestamps represent",
  "bag_name": "optional_reference_to_bag_file.mcap",
  "timestamps": [
    1760644685375549387,
    1760644695047075526,
    1760644717734949040
  ],
  "num_frames": 100,
  "sigma_seconds": 5.0,
  "min_gap_seconds": 0.5,
  "color_topic": "/BD03/d455/color/image_raw",
  "depth_topic": "/BD03/d455/depth/image_rect_raw",
  "max_time_diff_ms": 50.0,
  "notes": "Additional notes or context"
}
```

## Required Fields

- `timestamps`: Array of integers (timestamps in nanoseconds) - centers of Gaussian distributions

## Optional Fields

- `description`: Description of the configuration
- `bag_name`: Reference to the MCAP file (for documentation)
- `num_frames`: Number of frames to extract (default: 100)
- `sigma_seconds`: Gaussian standard deviation in seconds (default: 5.0)
  - Larger values = wider sampling around each timestamp
  - Smaller values = tighter focus on specific moments
- `min_gap_seconds`: **[Hard Sampling only]** Minimum time between frames in seconds (default: 0.5)
  - `0.5s`: Good balance of diversity and coverage
  - `1.0s`: More diversity, fewer similar frames
  - `0.2s`: Less strict, more frames but some redundancy
- `color_topic`: ROS topic for color images (default: `/BD03/d455/color/image_raw`)
- `depth_topic`: ROS topic for depth images (default: `/BD03/d455/depth/image_rect_raw`)
- `max_time_diff_ms`: Max time difference for color/depth matching in milliseconds (default: 50.0)
- `notes`: Additional notes

## Usage

### Hard Sampling (Recommended)

```bash
# Use a config file with hard sampling
python src/mcap_to_imgs_hard_sampling.py \
  data/bags/your_bag.mcap \
  --config config/gaussian_frame_sampling/your_config.json \
  -s 42

# Override minimum gap for more diversity
python src/mcap_to_imgs_hard_sampling.py \
  data/bags/your_bag.mcap \
  --config config/gaussian_frame_sampling/your_config.json \
  --min-gap 1.0 \
  -s 42

# Override number of frames
python src/mcap_to_imgs_hard_sampling.py \
  data/bags/your_bag.mcap \
  --config config/gaussian_frame_sampling/your_config.json \
  -n 300 \
  -s 42
```

### Gaussian Sampling (Original Method)

```bash
# Use a config file
python src/mcap_to_imgs_gaussian.py \
  data/bags/your_bag.mcap \
  --config config/gaussian_frame_sampling/your_config.json

# Override specific parameters
python src/mcap_to_imgs_gaussian.py \
  data/bags/your_bag.mcap \
  --config config/gaussian_frame_sampling/your_config.json \
  -n 200 \
  --sigma 10.0
```

## Why Hard Sampling?

### The Problem: Temporal Redundancy
- Frame at t=10.1s and t=10.2s are 99.9% identical
- Training on both wastes compute with minimal new information
- Leads to "event-level overfitting" - memorizing specific moments instead of learning general patterns

### The Solution: Hard Sampling
1. **Focus on important events** (via Gaussian distribution)
2. **Enforce minimum time gaps** (removes redundant frames)
3. **Use data augmentation during training** (for robustness)

### Comparison

| Method | Temporal Redundancy | Efficiency | Best For |
|--------|-------------------|------------|----------|
| **Gaussian** | High - many similar frames | Low - wasted compute | Quick exploration |
| **Hard Sampling** | Low - enforced gaps | High - diverse frames | Training datasets |

### Example Output

```
Initial sampling (oversampling by 3x): 750 frames
Enforcing minimum time gap of 0.5s between frames...
✓ After hard sampling: 310 frames (removed 440 temporally redundant frames)
```

You get fewer frames, but each one is meaningfully different!

## Command Line Args Override Config

All command line arguments take precedence over config file values, allowing you to:
- Test different parameters without editing configs
- Reuse configs across different experiments
- Quickly iterate on sampling strategies


