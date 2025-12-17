#!/usr/bin/env python3
"""
P&ID Image Slicing Utility

Dynamically slices high-resolution P&ID images into tiles based on image size.
Tiles are sized for optimal vision model processing with configurable overlap.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


def calculate_grid(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Calculate optimal grid dimensions based on image size."""
    cols = math.ceil(width / max_size)
    rows = math.ceil(height / max_size)
    return rows, cols


def slice_image(
    image_path: str,
    output_dir: str,
    max_size: int = 2048,
    overlap: float = 0.1
) -> dict:
    """
    Slice image into tiles with overlap.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save tiles
        max_size: Maximum tile dimension in pixels
        overlap: Overlap percentage (0.0 to 0.5)
    
    Returns:
        Metadata dictionary with tile information
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # For images that fit within max_size, create a single tile
    # This ensures consistent pipeline paths (tiles_metadata.json always exists)
    if width <= max_size and height <= max_size:
        print(f"Image {width}x{height} fits within {max_size}px. Creating single tile.")
        rows, cols = 1, 1
    else:
        rows, cols = calculate_grid(width, height, max_size)
        print(f"Image {width}x{height} -> {rows}x{cols} grid")
    
    # Calculate tile size with overlap
    base_tile_w = math.ceil(width / cols)
    base_tile_h = math.ceil(height / rows)
    overlap_w = int(base_tile_w * overlap)
    overlap_h = int(base_tile_h * overlap)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tiles = []
    tile_num = 0
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile coordinates with overlap
            x1 = max(0, col * base_tile_w - overlap_w)
            y1 = max(0, row * base_tile_h - overlap_h)
            x2 = min(width, (col + 1) * base_tile_w + overlap_w)
            y2 = min(height, (row + 1) * base_tile_h + overlap_h)
            
            # Crop and save tile
            tile = img.crop((x1, y1, x2, y2))
            tile_filename = f"tile_{row:02d}_{col:02d}.png"
            tile_path = output_path / tile_filename
            tile.save(tile_path, "PNG")
            
            # Normalized coordinates (0-1000 space)
            norm_x1 = int(x1 / width * 1000)
            norm_y1 = int(y1 / height * 1000)
            norm_x2 = int(x2 / width * 1000)
            norm_y2 = int(y2 / height * 1000)
            
            tiles.append({
                "tile_id": tile_num,
                "filename": tile_filename,
                "path": str(tile_path),
                "grid_position": {"row": row, "col": col},
                "pixel_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "normalized_bbox": {"x1": norm_x1, "y1": norm_y1, "x2": norm_x2, "y2": norm_y2},
                "size": {"width": x2 - x1, "height": y2 - y1}
            })
            
            tile_num += 1
            print(f"  Created tile {tile_num}/{rows*cols}: {tile_filename}")
    
    metadata = {
        "source_image": str(image_path),
        "original_size": {"width": width, "height": height},
        "sliced": True,
        "grid": {"rows": rows, "cols": cols},
        "max_tile_size": max_size,
        "overlap_percent": overlap,
        "total_tiles": len(tiles),
        "tiles": tiles
    }
    
    # Save metadata
    metadata_path = output_path / "tiles_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Slice P&ID images into tiles for vision model processing"
    )
    parser.add_argument("image_path", help="Path to input P&ID image")
    parser.add_argument(
        "--output-dir", "-o",
        default="./tiles",
        help="Output directory for tiles (default: ./tiles)"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=int,
        default=2048,
        help="Maximum tile dimension in pixels (default: 2048)"
    )
    parser.add_argument(
        "--overlap", "-v",
        type=float,
        default=0.1,
        help="Overlap percentage 0.0-0.5 (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    if not 0.0 <= args.overlap <= 0.5:
        print("Error: Overlap must be between 0.0 and 0.5")
        sys.exit(1)
    
    metadata = slice_image(
        args.image_path,
        args.output_dir,
        args.max_size,
        args.overlap
    )
    
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
