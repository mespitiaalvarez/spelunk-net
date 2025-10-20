#!/usr/bin/env python3
"""
Extract random color and depth image pairs from MCAP ROS bags (Random sampling).
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def decode_image(msg) -> np.ndarray:
    """
    Decode a ROS2 Image message to a numpy array.
    
    Args:
        msg: ROS2 sensor_msgs/msg/Image message
        
    Returns:
        numpy array of the image
    """
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    data = bytes(msg.data)  # Convert to bytes if needed
    
    # Convert bytes to numpy array based on encoding
    if encoding == "mono8" or encoding == "8UC1":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
    elif encoding == "mono16" or encoding == "16UC1":
        img = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
    elif encoding == "bgr8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    elif encoding == "rgb8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "bgra8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    elif encoding == "rgba8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    elif encoding == "32FC1":
        img = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    
    return img


def find_matching_pairs(color_msgs: List[Tuple], depth_msgs: List[Tuple], 
                       max_time_diff_ns: int = 50_000_000) -> List[Tuple[int, int]]:
    """
    Find matching color and depth message pairs based on timestamp proximity.
    
    Args:
        color_msgs: List of (timestamp, message) tuples for color images
        depth_msgs: List of (timestamp, message) tuples for depth images
        max_time_diff_ns: Maximum time difference in nanoseconds (default: 50ms)
        
    Returns:
        List of (color_idx, depth_idx) pairs
    """
    pairs = []
    depth_idx = 0
    
    for color_idx, (color_ts, _) in enumerate(color_msgs):
        # Find the closest depth message
        while depth_idx < len(depth_msgs) - 1:
            curr_diff = abs(depth_msgs[depth_idx][0] - color_ts)
            next_diff = abs(depth_msgs[depth_idx + 1][0] - color_ts)
            
            if next_diff < curr_diff:
                depth_idx += 1
            else:
                break
        
        # Check if the match is within the time threshold
        if depth_idx < len(depth_msgs):
            time_diff = abs(depth_msgs[depth_idx][0] - color_ts)
            if time_diff <= max_time_diff_ns:
                pairs.append((color_idx, depth_idx))
    
    return pairs


def extract_random_frames(mcap_path: str, 
                          color_topic: str, 
                          depth_topic: str,
                          num_frames: int,
                          output_base_dir: str = None,
                          max_time_diff_ms: float = 50.0,
                          seed: int = None):
    """
    Extract random color and depth image pairs from an MCAP file.
    
    Args:
        mcap_path: Path to the MCAP file
        color_topic: Topic name for color images
        depth_topic: Topic name for depth images
        num_frames: Number of random frames to extract
        output_base_dir: Base directory for output (default: data/processed_imgs)
        max_time_diff_ms: Maximum time difference between color and depth in milliseconds
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    mcap_path = Path(mcap_path)
    
    # Set default output base directory
    if output_base_dir is None:
        # Get the script directory and navigate to data/processed_imgs
        script_dir = Path(__file__).parent.parent
        output_base_dir = script_dir / "data" / "processed_imgs"
    else:
        output_base_dir = Path(output_base_dir)
    
    # Create a unique timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mcap_name = mcap_path.stem
    output_dir = output_base_dir / f"{mcap_name}_{timestamp}"
    
    # Create output directories
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading MCAP file: {mcap_path}")
    print(f"Color topic: {color_topic}")
    print(f"Depth topic: {depth_topic}")
    
    # Read all messages from both topics
    color_msgs = []
    depth_msgs = []
    
    # Create decoder factory for ROS2 messages
    decoder_factory = DecoderFactory()
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[decoder_factory])
        
        # Get summary info
        summary = reader.get_summary()
        if summary and summary.schemas:
            print(f"\nAvailable topics in bag:")
            topics_seen = set()
            for channel in summary.channels.values():
                if channel.topic not in topics_seen:
                    topics_seen.add(channel.topic)
                    schema = summary.schemas.get(channel.schema_id)
                    schema_name = schema.name if schema else "unknown"
                    print(f"  {channel.topic}: {schema_name}")
        
        print(f"\nReading messages...")
        
        # Read all messages
        for schema, channel, message, decoded_msg in tqdm(reader.iter_decoded_messages()):
            if channel.topic == color_topic:
                color_msgs.append((message.log_time, decoded_msg))
            elif channel.topic == depth_topic:
                depth_msgs.append((message.log_time, decoded_msg))
    
    print(f"\nFound {len(color_msgs)} color images and {len(depth_msgs)} depth images")
    
    if len(color_msgs) == 0 or len(depth_msgs) == 0:
        raise ValueError("No messages found on one or both topics")
    
    # Sort by timestamp
    color_msgs.sort(key=lambda x: x[0])
    depth_msgs.sort(key=lambda x: x[0])
    
    # Find matching pairs
    print("Finding matching color-depth pairs...")
    max_time_diff_ns = int(max_time_diff_ms * 1_000_000)
    pairs = find_matching_pairs(color_msgs, depth_msgs, max_time_diff_ns)
    
    print(f"Found {len(pairs)} matching pairs")
    
    if len(pairs) == 0:
        raise ValueError("No matching pairs found. Try increasing max_time_diff_ms")
    
    # Sample random pairs
    num_frames = min(num_frames, len(pairs))
    sampled_pairs = random.sample(pairs, num_frames)
    
    print(f"\nExtracting {num_frames} random frame pairs...")
    
    # Extract and save images
    for idx, (color_idx, depth_idx) in enumerate(tqdm(sampled_pairs)):
        color_ts, color_msg = color_msgs[color_idx]
        depth_ts, depth_msg = depth_msgs[depth_idx]
        
        # Decode images
        try:
            color_img = decode_image(color_msg)
            depth_img = decode_image(depth_msg)
            
            # Save images with timestamp-based filenames
            color_filename = color_dir / f"frame_{idx:06d}_{color_ts}.png"
            depth_filename = depth_dir / f"frame_{idx:06d}_{depth_ts}.png"
            
            cv2.imwrite(str(color_filename), color_img)
            
            # For depth images, save as 16-bit if possible
            if depth_img.dtype == np.uint16:
                cv2.imwrite(str(depth_filename), depth_img)
            elif depth_img.dtype == np.float32:
                # Convert float depth to uint16 (assuming meters, scale to mm)
                depth_mm = (depth_img * 1000).astype(np.uint16)
                cv2.imwrite(str(depth_filename), depth_mm)
            else:
                cv2.imwrite(str(depth_filename), depth_img)
            
        except Exception as e:
            print(f"\nError processing frame {idx}: {e}")
            continue
    
    print(f"\n✓ Done! Images saved to:")
    print(f"  Output directory: {output_dir}")
    print(f"  Color: {color_dir}")
    print(f"  Depth: {depth_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract random color and depth image pairs from MCAP ROS bags"
    )
    parser.add_argument(
        "mcap_path",
        type=str,
        help="Path to the MCAP file"
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=100,
        help="Number of random frames to extract (default: 100)"
    )
    parser.add_argument(
        "-o", "--output-base-dir",
        type=str,
        default=None,
        help="Base output directory (default: data/processed_imgs). A timestamped subdirectory will be created."
    )
    parser.add_argument(
        "-c", "--color-topic",
        type=str,
        default="/BD03/d455/color/image_raw",
        help="Color image topic (default: /BD03/d455/color/image_raw)"
    )
    parser.add_argument(
        "-d", "--depth-topic",
        type=str,
        default="/BD03/d455/depth/image_rect_raw",
        help="Depth image topic (default: /BD03/d455/depth/image_rect_raw)"
    )
    parser.add_argument(
        "-t", "--max-time-diff",
        type=float,
        default=50.0,
        help="Maximum time difference between color and depth in milliseconds (default: 50.0)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    extract_random_frames(
        mcap_path=args.mcap_path,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        num_frames=args.num_frames,
        output_base_dir=args.output_base_dir,
        max_time_diff_ms=args.max_time_diff,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
