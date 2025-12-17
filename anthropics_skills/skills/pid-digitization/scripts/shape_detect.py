#!/usr/bin/env python3
from __future__ import annotations

"""
CV-Based Shape Detection for P&ID Diagrams

Implements basic shape detection for P&ID symbols:
1. Hough circles for instrument bubbles (ISA-5.1 field/control room instruments)
2. Rectangle detection for control panels and shared displays
3. Output normalized 0-1000 coordinates as bbox hints

These are "proposals" for the VLM - geometric hints to guide symbol detection.

Install dependencies:
    pip install opencv-python numpy pillow
"""

import argparse
import json
import sys
from pathlib import Path

_missing_cv_deps = None
_missing_pillow = None

try:
    import cv2
    import numpy as np
except ImportError as e:
    _missing_cv_deps = e
    cv2 = None
    np = None

try:
    from PIL import Image
except ImportError as e:
    _missing_pillow = e
    Image = None


def _ensure_dependencies():
    if _missing_cv_deps is not None:
        raise ImportError(
            "Missing dependency. Install with: pip install opencv-python numpy"
        ) from _missing_cv_deps
    if _missing_pillow is not None:
        raise ImportError("Pillow is required. Install with: pip install Pillow") from _missing_pillow


def detect_circles(gray: np.ndarray, 
                   min_radius: int = 10, 
                   max_radius: int = 50,
                   param1: int = 50,
                   param2: int = 30) -> list:
    """
    Detect circles using Hough Circle Transform.
    
    In P&IDs, circles typically represent:
    - Instrument bubbles (ISA-5.1)
    - Control room instruments (circle with line)
    - Pumps (circle with arrow)
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,  # Minimum distance between circle centers
        param1=param1,  # Canny edge detection threshold
        param2=param2,  # Accumulator threshold for circle centers
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detected = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            x, y, r = circle
            detected.append({
                "center": (int(x), int(y)),
                "radius": int(r),
                "bbox_pixels": [int(y - r), int(x - r), int(y + r), int(x + r)]  # ymin, xmin, ymax, xmax
            })
    
    return detected


def detect_rectangles(gray: np.ndarray,
                      min_area: int = 500,
                      max_area: int = 50000,
                      aspect_ratio_range: tuple = (0.5, 2.0)) -> list:
    """
    Detect rectangles for control panels, shared displays, and equipment symbols.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a rectangle (4 vertices)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    rectangles.append({
                        "bbox_pixels": [y, x, y + h, x + w],  # ymin, xmin, ymax, xmax
                        "area": area,
                        "aspect_ratio": round(aspect_ratio, 2)
                    })
    
    return rectangles


def classify_circle_type(circle: dict, gray: np.ndarray) -> str:
    """
    Attempt to classify circle type based on internal features.
    
    - Empty circle: Field instrument
    - Circle with horizontal line: Control room instrument
    - Circle with text: Tagged instrument
    """
    x, y = circle["center"]
    r = circle["radius"]
    
    # Extract ROI
    y1, y2 = max(0, y - r), min(gray.shape[0], y + r)
    x1, x2 = max(0, x - r), min(gray.shape[1], x + r)
    
    if y2 <= y1 or x2 <= x1:
        return "InstrumentBubble"
    
    roi = gray[y1:y2, x1:x2]
    
    # Check for horizontal line (control room indicator)
    horizontal_sum = np.sum(roi[roi.shape[0]//2-2:roi.shape[0]//2+2, :])
    total_sum = np.sum(roi) if np.sum(roi) > 0 else 1
    
    if horizontal_sum / total_sum > 0.3:
        return "ControlRoomInstrument"
    
    return "FieldInstrument"


def normalize_bbox(bbox_pixels: list, width: int, height: int) -> list:
    """Normalize bbox to 0-1000 space."""
    ymin, xmin, ymax, xmax = bbox_pixels
    return [
        int((ymin / height) * 1000),
        int((xmin / width) * 1000),
        int((ymax / height) * 1000),
        int((xmax / width) * 1000)
    ]


def detect_shapes(image_path: str, output_dir: str = None,
                  min_circle_radius: int = 10,
                  max_circle_radius: int = 50,
                  detect_rectangles_flag: bool = True) -> dict:
    """
    Main shape detection function.
    
    Returns dict with:
    - circles: Instrument bubble candidates
    - rectangles: Panel/equipment candidates
    - metadata: Detection parameters
    """
    _ensure_dependencies()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Processing image: {width}x{height}")
    
    # Detect circles (instrument bubbles)
    print("  Detecting circles (instrument bubbles)...")
    circles = detect_circles(gray, min_circle_radius, max_circle_radius)
    
    # Classify and normalize
    circle_candidates = []
    for i, circle in enumerate(circles):
        circle_type = classify_circle_type(circle, gray)
        circle_candidates.append({
            "id": f"CV-SHAPE-{i+1:03d}",
            "type": circle_type,
            "bbox": normalize_bbox(circle["bbox_pixels"], width, height),
            "bbox_pixels": circle["bbox_pixels"],
            "center_normalized": [
                int((circle["center"][0] / width) * 1000),
                int((circle["center"][1] / height) * 1000)
            ],
            "radius_normalized": int((circle["radius"] / min(width, height)) * 1000),
            "source": "cv_hough_circle"
        })
    
    print(f"    Found {len(circle_candidates)} circle candidates")
    
    # Detect rectangles
    rect_candidates = []
    if detect_rectangles_flag:
        print("  Detecting rectangles...")
        rectangles = detect_rectangles(gray)
        
        for i, rect in enumerate(rectangles):
            rect_candidates.append({
                "id": f"CV-RECT-{i+1:03d}",
                "type": "RectangularSymbol",
                "bbox": normalize_bbox(rect["bbox_pixels"], width, height),
                "bbox_pixels": rect["bbox_pixels"],
                "aspect_ratio": rect["aspect_ratio"],
                "source": "cv_contour"
            })
        
        print(f"    Found {len(rect_candidates)} rectangle candidates")
    
    # Build output
    result = {
        "source_image": str(image_path),
        "image_size": {"width": width, "height": height},
        "coordinate_system": "normalized_0_1000",
        "detection_params": {
            "min_circle_radius": min_circle_radius,
            "max_circle_radius": max_circle_radius,
            "detect_rectangles": detect_rectangles_flag
        },
        "circles": circle_candidates,
        "rectangles": rect_candidates,
        "summary": {
            "total_circles": len(circle_candidates),
            "total_rectangles": len(rect_candidates),
            "field_instruments": len([c for c in circle_candidates if c["type"] == "FieldInstrument"]),
            "control_room_instruments": len([c for c in circle_candidates if c["type"] == "ControlRoomInstrument"])
        }
    }
    
    # Save output if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "cv_shapes.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput saved to: {output_file}")
    
    return result


def main():
    try:
        _ensure_dependencies()
    except ImportError as e:
        print(f"Error: {e}")
        details = e.__cause__
        if details is not None:
            print(f"Details: {details}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CV-based shape detection for P&ID diagrams"
    )
    parser.add_argument("image_path", help="Path to P&ID image")
    parser.add_argument(
        "--output-dir", "-o",
        default="./cv_output",
        help="Output directory (default: ./cv_output)"
    )
    parser.add_argument(
        "--min-radius",
        type=int, default=10,
        help="Minimum circle radius in pixels (default: 10)"
    )
    parser.add_argument(
        "--max-radius",
        type=int, default=50,
        help="Maximum circle radius in pixels (default: 50)"
    )
    parser.add_argument(
        "--no-rectangles",
        action="store_true",
        help="Skip rectangle detection"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    result = detect_shapes(
        args.image_path,
        args.output_dir,
        args.min_radius,
        args.max_radius,
        detect_rectangles_flag=not args.no_rectangles
    )
    
    print(f"\n{'='*50}")
    print("SHAPE DETECTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Circles (instrument candidates): {result['summary']['total_circles']}")
    print(f"    - Field instruments: {result['summary']['field_instruments']}")
    print(f"    - Control room: {result['summary']['control_room_instruments']}")
    print(f"  Rectangles (panel candidates): {result['summary']['total_rectangles']}")


if __name__ == "__main__":
    main()
