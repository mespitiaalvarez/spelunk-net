# Timestamp Configuration Files

This directory contains JSON configuration files for specifying timestamps and parameters for Gaussian-weighted sampling.

## Format

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
  "color_topic": "/BD03/d455/color/image_raw",
  "depth_topic": "/BD03/d455/depth/image_rect_raw",
  "max_time_diff_ms": 50.0,
  "notes": "Additional notes or context"
}
```

## Required Fields

- `timestamps`: Array of integers (timestamps in nanoseconds)

## Optional Fields

- `description`: Description of the configuration
- `bag_name`: Reference to the MCAP file (for documentation)
- `num_frames`: Number of frames to extract (default: 100)
- `sigma_seconds`: Gaussian standard deviation in seconds (default: 5.0)
- `color_topic`: ROS topic for color images
- `depth_topic`: ROS topic for depth images
- `max_time_diff_ms`: Max time difference for color/depth matching in milliseconds
- `notes`: Additional notes

## Usage

```bash
# Use a config file
python3 src/mcap_to_imgs_gaussian.py data/bags/your_bag.mcap --config config/your_config.json

# Override specific parameters
python3 src/mcap_to_imgs_gaussian.py data/bags/your_bag.mcap --config config/your_config.json -n 200

# Command line args always take precedence over config values
python3 src/mcap_to_imgs_gaussian.py data/bags/your_bag.mcap --config config/your_config.json --sigma 10.0
```


