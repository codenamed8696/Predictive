#!/usr/bin/env python3
"""
P&ID Visual Debugging Tool

Overlays extracted bounding boxes onto the original P&ID image for visual validation.
Color coding:
  - Equipment: Blue
  - Instruments: Green  
  - Valves: Red
  - Lines: Yellow (waypoints)
  - OCR Text: Cyan

Usage:
    python visualize_results.py <image_path> <json_path> [options]
    
Install dependencies:
    pip install pillow
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# Color scheme for different entity types
COLORS = {
    "equipment": (0, 100, 255),      # Blue
    "instruments": (0, 200, 100),    # Green
    "valves": (255, 80, 80),         # Red
    "lines": (255, 200, 0),          # Yellow
    "ocr": (0, 220, 220),            # Cyan
    "waypoint": (255, 150, 0),       # Orange
}


def denormalize_bbox(bbox: list, width: int, height: int) -> tuple:
    """
    Convert normalized 0-1000 bbox to pixel coordinates.
    bbox format: [ymin, xmin, ymax, xmax]
    """
    ymin, xmin, ymax, xmax = bbox
    return (
        int(xmin * width / 1000),
        int(ymin * height / 1000),
        int(xmax * width / 1000),
        int(ymax * height / 1000)
    )


def denormalize_point(point: list, width: int, height: int) -> tuple:
    """Convert normalized 0-1000 point to pixel coordinates."""
    x, y = point
    return (int(x * width / 1000), int(y * height / 1000))


def draw_entity(draw: ImageDraw, bbox: list, label: str, color: tuple, 
                width: int, height: int, line_width: int = 2):
    """Draw a labeled bounding box."""
    px_bbox = denormalize_bbox(bbox, width, height)
    
    # Draw rectangle
    draw.rectangle(px_bbox, outline=color, width=line_width)
    
    # Draw label background
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    label_x = px_bbox[0]
    label_y = max(0, px_bbox[1] - text_height - 4)
    
    draw.rectangle(
        [label_x, label_y, label_x + text_width + 4, label_y + text_height + 2],
        fill=color
    )
    draw.text((label_x + 2, label_y), label, fill=(255, 255, 255))


def draw_waypoints(draw: ImageDraw, waypoints: list, color: tuple,
                   width: int, height: int, line_width: int = 2):
    """Draw line waypoints as connected segments."""
    if len(waypoints) < 2:
        return
    
    px_points = [denormalize_point(p, width, height) for p in waypoints]
    
    # Draw line segments
    for i in range(len(px_points) - 1):
        draw.line([px_points[i], px_points[i + 1]], fill=color, width=line_width)
    
    # Draw waypoint markers
    for p in px_points:
        r = 3
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], 
                     fill=COLORS["waypoint"], outline=(0, 0, 0))


def visualize(image_path: str, json_path: str, output_path: str = None,
              show_equipment: bool = True, show_instruments: bool = True,
              show_valves: bool = True, show_lines: bool = True,
              show_ocr: bool = False, min_confidence: float = 0.0):
    """
    Create visualization overlay on P&ID image.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Create overlay with transparency
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    # Draw equipment
    if show_equipment:
        for eq in data.get("nodes", {}).get("equipment", []):
            if eq.get("confidence", 1.0) < min_confidence:
                continue
            bbox = eq.get("bbox") or eq.get("normalized_bbox")
            if bbox:
                label = f"{eq.get('tag', '?')} ({eq.get('class', '?')})"
                draw_entity(draw, bbox, label, COLORS["equipment"], width, height)
    
    # Draw instruments
    if show_instruments:
        for inst in data.get("nodes", {}).get("instruments", []):
            if inst.get("confidence", 1.0) < min_confidence:
                continue
            bbox = inst.get("bbox") or inst.get("normalized_bbox")
            if bbox:
                label = f"{inst.get('tag', '?')} [{inst.get('isa_code', inst.get('function_letters', '?'))}]"
                draw_entity(draw, bbox, label, COLORS["instruments"], width, height)
    
    # Draw valves
    if show_valves:
        for valve in data.get("nodes", {}).get("valves", []):
            if valve.get("confidence", 1.0) < min_confidence:
                continue
            bbox = valve.get("bbox") or valve.get("normalized_bbox")
            if bbox:
                label = f"{valve.get('tag', '?')} ({valve.get('class', valve.get('type', '?'))})"
                draw_entity(draw, bbox, label, COLORS["valves"], width, height)
    
    # Draw lines (waypoints)
    if show_lines:
        for line in data.get("edges", {}).get("process_lines", []):
            waypoints = line.get("route_waypoints", line.get("waypoints", []))
            if waypoints:
                draw_waypoints(draw, waypoints, COLORS["lines"], width, height)
    
    # Draw OCR extractions if available
    if show_ocr:
        ocr_data = data.get("all_extractions", data.get("detections", {}).get("all", []))
        if isinstance(ocr_data, dict):
            # Flatten grouped OCR
            ocr_data = []
            for group in data.get("detections", {}).values():
                ocr_data.extend(group)
        
        for ocr in ocr_data:
            if ocr.get("confidence", 1.0) < min_confidence:
                continue
            bbox = ocr.get("bbox") or ocr.get("bbox_global")
            if bbox:
                label = ocr.get("text", "?")[:20]  # Truncate long text
                draw_entity(draw, bbox, label, COLORS["ocr"], width, height, line_width=1)
    
    # Composite overlay onto original
    img = img.convert("RGBA")
    result = Image.alpha_composite(img, overlay)
    result = result.convert("RGB")
    
    # Save or show
    if output_path:
        result.save(output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        result.show()
    
    # Print summary
    nodes = data.get("nodes", {})
    print(f"\nVisualization Summary:")
    print(f"  Equipment: {len(nodes.get('equipment', []))}")
    print(f"  Instruments: {len(nodes.get('instruments', []))}")
    print(f"  Valves: {len(nodes.get('valves', []))}")
    print(f"  Lines: {len(data.get('edges', {}).get('process_lines', []))}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Visualize P&ID extraction results by overlaying bboxes on the image"
    )
    parser.add_argument("image_path", help="Path to original P&ID image")
    parser.add_argument("json_path", help="Path to extraction JSON result")
    parser.add_argument(
        "--output", "-o",
        help="Output image path (if not specified, displays in viewer)"
    )
    parser.add_argument(
        "--min-confidence", "-c",
        type=float, default=0.0,
        help="Minimum confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--no-equipment", action="store_true",
        help="Hide equipment boxes"
    )
    parser.add_argument(
        "--no-instruments", action="store_true",
        help="Hide instrument boxes"
    )
    parser.add_argument(
        "--no-valves", action="store_true",
        help="Hide valve boxes"
    )
    parser.add_argument(
        "--no-lines", action="store_true",
        help="Hide line waypoints"
    )
    parser.add_argument(
        "--show-ocr", action="store_true",
        help="Show OCR text extractions (if present in JSON)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    if not Path(args.json_path).exists():
        print(f"Error: JSON not found: {args.json_path}")
        sys.exit(1)
    
    visualize(
        args.image_path,
        args.json_path,
        args.output,
        show_equipment=not args.no_equipment,
        show_instruments=not args.no_instruments,
        show_valves=not args.no_valves,
        show_lines=not args.no_lines,
        show_ocr=args.show_ocr,
        min_confidence=args.min_confidence
    )


if __name__ == "__main__":
    main()
