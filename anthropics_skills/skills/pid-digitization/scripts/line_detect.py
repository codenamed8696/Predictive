#!/usr/bin/env python3
from __future__ import annotations

"""
CV-Based Line Detection for P&ID Diagrams

Implements morphological line detection as described in Digitize-PID paper:
1. Horizontal/vertical kernel erosion-dilation for solid lines
2. Dashed-line detection via segment length/gap analysis + DBSCAN clustering
3. Centerline extraction via skeletonization
4. Output normalized 0-1000 coordinates

Outputs:
- Solid lines (process_lines candidates) with waypoints
- Dashed lines (signal_lines candidates) with waypoints

Install dependencies:
    pip install opencv-python numpy scikit-learn pillow
"""

import argparse
import json
import math
import sys
from pathlib import Path

_missing_cv_deps = None
_missing_pillow = None

try:
    import cv2
    import numpy as np
    from sklearn.cluster import DBSCAN
except ImportError as e:
    _missing_cv_deps = e
    cv2 = None
    np = None
    DBSCAN = None

try:
    from PIL import Image
except ImportError as e:
    _missing_pillow = e
    Image = None


def _ensure_dependencies():
    if _missing_cv_deps is not None:
        raise ImportError(
            "Missing dependency. Install with: pip install opencv-python numpy scikit-learn"
        ) from _missing_cv_deps
    if _missing_pillow is not None:
        raise ImportError("Pillow is required. Install with: pip install Pillow") from _missing_pillow


def detect_solid_lines(gray: np.ndarray, kernel_length: int = 40) -> np.ndarray:
    """
    Detect solid lines using morphological operations.
    
    Uses horizontal and vertical kernels to extract line structures.
    """
    # Horizontal kernel for horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Vertical kernel for vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Combine horizontal and vertical
    combined = cv2.bitwise_or(h_lines, v_lines)
    
    return combined


def extract_centerlines(binary: np.ndarray) -> np.ndarray:
    """
    Extract centerlines using morphological skeletonization.
    """
    # Ensure binary
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonization
    skeleton = np.zeros(binary.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    return skeleton


def detect_line_segments(skeleton: np.ndarray, min_length: int = 20) -> list:
    """
    Detect line segments from skeleton using probabilistic Hough transform.
    
    Returns list of segments: [(x1, y1, x2, y2), ...]
    """
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=10
    )
    
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append((x1, y1, x2, y2))
    
    return segments


def classify_segment_orientation(x1: int, y1: int, x2: int, y2: int, tolerance_deg: float = 15.0) -> str:
    """
    Classify segment as horizontal, vertical, or diagonal.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx == 0 and dy == 0:
        return "point"
    
    angle = math.degrees(math.atan2(dy, dx))
    
    if angle <= tolerance_deg:
        return "horizontal"
    elif angle >= (90 - tolerance_deg):
        return "vertical"
    else:
        return "diagonal"


def detect_dashed_lines(gray: np.ndarray, 
                        min_segment_length: int = 5,
                        max_segment_length: int = 30,
                        max_gap: int = 20,
                        eps: float = 15.0,
                        min_samples: int = 3) -> list:
    """
    Detect dashed lines using segment analysis + DBSCAN clustering.
    
    Paper approach:
    1. Find small line segments within length thresholds
    2. Cluster collinear segments using DBSCAN
    3. Merge clusters into dashed-line paths
    """
    # Edge detection for small segments
    edges = cv2.Canny(gray, 50, 150)
    
    # Find short line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=min_segment_length,
        maxLineGap=5
    )
    
    if lines is None:
        return []
    
    # Filter by segment length
    dash_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if min_segment_length <= length <= max_segment_length:
            # Calculate midpoint and angle
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            dash_candidates.append({
                "segment": (x1, y1, x2, y2),
                "midpoint": (mx, my),
                "angle": angle,
                "length": length
            })
    
    if len(dash_candidates) < min_samples:
        return []
    
    # Cluster by position and angle using DBSCAN
    # Features: midpoint_x, midpoint_y, angle (scaled)
    features = np.array([
        [d["midpoint"][0], d["midpoint"][1], d["angle"] * 100]  # Scale angle
        for d in dash_candidates
    ])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    
    # Group segments by cluster
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:
            continue  # Noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(dash_candidates[i])
    
    # Build dashed line paths from clusters
    dashed_lines = []
    for label, segments in clusters.items():
        if len(segments) < min_samples:
            continue
        
        # Sort segments by position along their primary axis
        avg_angle = np.mean([s["angle"] for s in segments])
        
        if abs(avg_angle) < math.pi / 4:
            # More horizontal - sort by x
            segments.sort(key=lambda s: s["midpoint"][0])
        else:
            # More vertical - sort by y
            segments.sort(key=lambda s: s["midpoint"][1])
        
        # Extract waypoints (endpoints of first and last segments)
        first = segments[0]["segment"]
        last = segments[-1]["segment"]
        
        waypoints = [(first[0], first[1])]
        for seg in segments:
            waypoints.append((seg["segment"][2], seg["segment"][3]))
        
        dashed_lines.append({
            "waypoints": waypoints,
            "segment_count": len(segments),
            "avg_segment_length": np.mean([s["length"] for s in segments])
        })
    
    return dashed_lines


def normalize_coordinates(value: float, max_value: int) -> int:
    """Normalize to 0-1000 space."""
    return int((value / max_value) * 1000)


def segments_to_waypoints(segments: list, width: int, height: int) -> list:
    """
    Convert line segments to waypoint paths with normalized coordinates.
    Groups connected segments into continuous paths.
    """
    if not segments:
        return []
    
    # Build connectivity graph
    endpoints = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        endpoints.append({"point": (x1, y1), "segment_idx": i, "end": 0})
        endpoints.append({"point": (x2, y2), "segment_idx": i, "end": 1})
    
    # Cluster nearby endpoints (junction detection)
    if len(endpoints) < 2:
        # Single segment
        x1, y1, x2, y2 = segments[0]
        return [[
            [normalize_coordinates(x1, width), normalize_coordinates(y1, height)],
            [normalize_coordinates(x2, width), normalize_coordinates(y2, height)]
        ]]
    
    points = np.array([e["point"] for e in endpoints])
    clustering = DBSCAN(eps=10.0, min_samples=1).fit(points)
    
    # Group paths by following connections
    paths = []
    visited_segments = set()
    
    for seg_idx, (x1, y1, x2, y2) in enumerate(segments):
        if seg_idx in visited_segments:
            continue
        
        visited_segments.add(seg_idx)
        path = [
            [normalize_coordinates(x1, width), normalize_coordinates(y1, height)],
            [normalize_coordinates(x2, width), normalize_coordinates(y2, height)]
        ]
        paths.append(path)
    
    return paths


def detect_lines(image_path: str, output_dir: str = None, 
                 kernel_length: int = 40,
                 min_line_length: int = 20,
                 detect_dashed: bool = True) -> dict:
    """
    Main line detection function.
    
    Returns dict with:
    - solid_lines: process line candidates with waypoints
    - dashed_lines: signal line candidates with waypoints
    - metadata: detection parameters and stats
    """
    _ensure_dependencies()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert if mostly white (P&ID standard)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Processing image: {width}x{height}")
    
    # Detect solid lines
    print("  Detecting solid lines...")
    solid_mask = detect_solid_lines(binary, kernel_length)
    skeleton = extract_centerlines(solid_mask)
    solid_segments = detect_line_segments(skeleton, min_line_length)
    solid_paths = segments_to_waypoints(solid_segments, width, height)
    
    print(f"    Found {len(solid_segments)} segments -> {len(solid_paths)} paths")
    
    # Detect dashed lines
    dashed_paths = []
    if detect_dashed:
        print("  Detecting dashed lines...")
        dashed_results = detect_dashed_lines(gray)
        
        for dash in dashed_results:
            # Normalize waypoints
            norm_waypoints = [
                [normalize_coordinates(x, width), normalize_coordinates(y, height)]
                for x, y in dash["waypoints"]
            ]
            dashed_paths.append({
                "waypoints": norm_waypoints,
                "segment_count": dash["segment_count"]
            })
        
        print(f"    Found {len(dashed_paths)} dashed line clusters")
    
    # Build output
    result = {
        "source_image": str(image_path),
        "image_size": {"width": width, "height": height},
        "coordinate_system": "normalized_0_1000",
        "detection_params": {
            "kernel_length": kernel_length,
            "min_line_length": min_line_length,
            "detect_dashed": detect_dashed
        },
        "solid_lines": [
            {
                "id": f"CV-LINE-{i+1:03d}",
                "line_type": "Process_Candidate",
                "route_waypoints": path,
                "source": "cv_detection"
            }
            for i, path in enumerate(solid_paths)
        ],
        "dashed_lines": [
            {
                "id": f"CV-SIG-{i+1:03d}",
                "line_type": "Signal_Candidate",
                "route_waypoints": dash["waypoints"],
                "segment_count": dash["segment_count"],
                "source": "cv_detection"
            }
            for i, dash in enumerate(dashed_paths)
        ],
        "summary": {
            "total_solid_lines": len(solid_paths),
            "total_dashed_lines": len(dashed_paths)
        }
    }
    
    # Save output if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "cv_lines.json"
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
        description="CV-based line detection for P&ID diagrams"
    )
    parser.add_argument("image_path", help="Path to P&ID image")
    parser.add_argument(
        "--output-dir", "-o",
        default="./cv_output",
        help="Output directory (default: ./cv_output)"
    )
    parser.add_argument(
        "--kernel-length", "-k",
        type=int, default=40,
        help="Morphological kernel length (default: 40)"
    )
    parser.add_argument(
        "--min-line-length", "-l",
        type=int, default=20,
        help="Minimum line segment length (default: 20)"
    )
    parser.add_argument(
        "--no-dashed",
        action="store_true",
        help="Skip dashed line detection"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    result = detect_lines(
        args.image_path,
        args.output_dir,
        args.kernel_length,
        args.min_line_length,
        detect_dashed=not args.no_dashed
    )
    
    print(f"\n{'='*50}")
    print("LINE DETECTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Solid lines (process candidates): {result['summary']['total_solid_lines']}")
    print(f"  Dashed lines (signal candidates): {result['summary']['total_dashed_lines']}")


if __name__ == "__main__":
    main()
