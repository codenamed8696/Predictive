#!/usr/bin/env python3
"""
OCR Pre-Processing for P&ID Diagrams

Uses PaddleOCR (recommended) or EasyOCR to extract text labels from P&ID images.
Follows the same slicing pattern as the vision workflow for consistency.

OUTPUT MODES:
1. Per-tile OCR: Each tile gets its own JSON with TILE-RELATIVE 0-1000 coordinates
   (Use this when feeding OCR to vision tool processing individual tiles)
2. Global OCR: All detections in one JSON with GLOBAL 0-1000 coordinates
   (Use this for final merge step)

Install dependencies:
    pip install paddleocr paddlepaddle pillow  # Recommended
    # OR
    pip install easyocr pillow  # Alternative
"""

import argparse
import json
import math
import re
import sys
import os
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

# Try to import OCR libraries
OCR_ENGINE = None

try:
    from paddleocr import PaddleOCR
    OCR_ENGINE = "paddleocr"
except ImportError:
    try:
        import easyocr
        OCR_ENGINE = "easyocr"
    except ImportError:
        pass


def classify_text(text: str) -> str:
    """
    Classify detected text by P&ID nomenclature patterns.
    
    ISA-5.1 Standard Patterns:
    - Equipment: [A-Z]{1,2}-\d{2,4}[A-Z]? (e.g., T-101, P-102A, TK-100)
    - Instrument: [A-Z]{2,4}-\d{2,4} (e.g., TIC-101, PT-200, FV-300)
    - Valve: [HF]?V-?\d{2,4} or control valves like FV-XXX, TV-XXX
    - Line Number: \d{1,2}"-[A-Z]{2,4}-\d+ (e.g., 4"-PA-001)
    """
    text = text.strip().upper()
    
    # Skip very short or very long strings
    if len(text) < 2 or len(text) > 20:
        return "label"
    
    # Line number: 4"-PA-001, 2"-CW-050, 1.5"-SS-100
    # Must check first - contains equipment-like patterns
    if re.match(r'^\d{1,2}\.?\d?["\'-]?[A-Z]{2,4}[-]?\d{2,4}', text):
        return "line_number"
    
    # Instrument tag (ISA-5.1): TIC-101, PT-200, FV-300, LSHH-100
    # First 2-4 letters are function codes, followed by loop number
    # Common patterns: PI, TI, FI, LI, PT, TT, FT, LT, TIC, PIC, FIC, LIC
    if re.match(r'^[AFHLPSTQVW][A-Z]{1,3}-?\d{2,5}$', text):
        return "instrument_tag"
    
    # Control valve: FV-XXX, TV-XXX, LV-XXX, PV-XXX (final control elements)
    if re.match(r'^[AFHLPT]V-?\d{2,4}$', text):
        return "valve_tag"
    
    # Manual valve: V-101, HV-100, BV-50
    if re.match(r'^[BHM]?V-?\d{2,4}[A-Z]?$', text):
        return "valve_tag"
    
    # Equipment tag: T-101, P-102A, V-50, TK-100, E-186, R-200
    # Single or double letter prefix, hyphen, 2-4 digits, optional suffix
    if re.match(r'^[A-Z]{1,2}-\d{2,4}[A-Z]?$', text):
        return "equipment_tag"
    
    # Equipment with longer prefix: TK-228A, HX-100
    if re.match(r'^(TK|HX|RX|CV|FD)-?\d{2,4}[A-Z]?$', text):
        return "equipment_tag"
    
    # Pressure/temp/flow values: 150 PSI, 200°C, 100 GPM
    if re.match(r'^\d+\.?\d*\s*(PSI|PSIG|PSIA|BAR|BARG|°?[CF]|MM|IN|GPM|LPM|M3/H)', text):
        return "measurement"
    
    # Percentage values: 50%, 100%
    if re.match(r'^\d+\.?\d*\s*%$', text):
        return "measurement"
    
    return "label"


def normalize_to_tile_relative(bbox_pixels: list, tile_width: int, tile_height: int) -> list:
    """
    Normalize bounding box to 0-1000 scale RELATIVE TO THE TILE.
    Use this for per-tile output that will be fed to vision tool.
    """
    ymin, xmin, ymax, xmax = bbox_pixels
    
    norm_ymin = int((ymin / tile_height) * 1000)
    norm_xmin = int((xmin / tile_width) * 1000)
    norm_ymax = int((ymax / tile_height) * 1000)
    norm_xmax = int((xmax / tile_width) * 1000)
    
    return [norm_ymin, norm_xmin, norm_ymax, norm_xmax]


def normalize_to_global(bbox_pixels: list, tile_width: int, tile_height: int,
                        tile_offset_x: int, tile_offset_y: int,
                        global_width: int, global_height: int) -> list:
    """
    Normalize bounding box to 0-1000 scale RELATIVE TO FULL IMAGE.
    Use this for global output that will be used in merge step.
    """
    ymin, xmin, ymax, xmax = bbox_pixels
    
    # Convert to global pixel coordinates
    global_xmin = xmin + tile_offset_x
    global_xmax = xmax + tile_offset_x
    global_ymin = ymin + tile_offset_y
    global_ymax = ymax + tile_offset_y
    
    # Normalize to 0-1000 scale
    norm_ymin = int((global_ymin / global_height) * 1000)
    norm_xmin = int((global_xmin / global_width) * 1000)
    norm_ymax = int((global_ymax / global_height) * 1000)
    norm_xmax = int((global_xmax / global_width) * 1000)
    
    return [norm_ymin, norm_xmin, norm_ymax, norm_xmax]


def extract_with_paddleocr(image_path: str) -> tuple:
    """Extract text using PaddleOCR."""
    # Note: use_textline_orientation replaces deprecated use_angle_cls
    # show_log and cls parameters removed in newer PaddleOCR versions
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    result = ocr.ocr(image_path)
    
    img = Image.open(image_path)
    width, height = img.size
    
    extractions = []
    
    if not result:
        return extractions, width, height
    
    # Handle new PaddleX format (result is list of inference objects)
    # vs old format (result is list of lists of [bbox, (text, score)] tuples)
    for page_result in result:
        if page_result is None:
            continue
        
        # Check if this is a PaddleX OCRResult (dict-like object)
        if hasattr(page_result, 'keys') and callable(page_result.keys):
            # New PaddleX format: OCRResult is dict-like
            # Keys are: rec_texts, rec_scores, dt_polys
            try:
                # Use correct plural key names for PaddleX
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                dt_polys = page_result.get('dt_polys', [])
                
                if rec_texts and dt_polys:
                    for i, poly in enumerate(dt_polys):
                        text = rec_texts[i] if i < len(rec_texts) else ""
                        score = rec_scores[i] if i < len(rec_scores) else 0.5
                        
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        
                        extractions.append({
                            "text": str(text),
                            "bbox_pixels": [min(y_coords), min(x_coords), max(y_coords), max(x_coords)],
                            "confidence": round(float(score), 3),
                        })
            except Exception as e:
                # Skip errors in parsing
                continue
        else:
            # Old list-based format: [[bbox, (text, score)], ...]
            for line in page_result:
                try:
                    bbox = line[0]
                    
                    if isinstance(line[1], dict):
                        text = line[1].get('text', '')
                        confidence = line[1].get('score', 0.0)
                    elif isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                    else:
                        text = str(line[1])
                        confidence = 0.5
                    
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    extractions.append({
                        "text": text,
                        "bbox_pixels": [min(y_coords), min(x_coords), max(y_coords), max(x_coords)],
                        "confidence": round(float(confidence), 3),
                    })
                except (IndexError, TypeError, KeyError):
                    continue
    
    return extractions, width, height


def extract_with_easyocr(image_path: str) -> tuple:
    """Extract text using EasyOCR."""
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    
    img = Image.open(image_path)
    width, height = img.size
    
    extractions = []
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]
        
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        
        extractions.append({
            "text": text,
            "bbox_pixels": [min(y_coords), min(x_coords), max(y_coords), max(x_coords)],
            "confidence": round(confidence, 3),
        })
    
    return extractions, width, height


def calculate_grid(width: int, height: int, max_size: int) -> tuple:
    """Calculate optimal grid dimensions."""
    cols = math.ceil(width / max_size)
    rows = math.ceil(height / max_size)
    return rows, cols


def process_image(image_path: str, output_dir: str, max_size: int = 2048, overlap: float = 0.1) -> dict:
    """
    Process image with OCR, following slicing pattern if needed.
    
    Outputs:
    1. Per-tile JSON files with TILE-RELATIVE coordinates (for vision tool)
    2. Global summary with all detections in GLOBAL coordinates (for merge step)
    """
    if OCR_ENGINE is None:
        print("Error: No OCR engine available.")
        print("Install: pip install paddleocr paddlepaddle")
        sys.exit(1)
    
    img = Image.open(image_path)
    global_width, global_height = img.size
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_global_extractions = []
    tile_outputs = []
    
    # Always use tiling approach for consistent output paths
    # Small images get 1x1 grid (single tile)
    if global_width <= max_size and global_height <= max_size:
        rows, cols = 1, 1
        print(f"Image {global_width}x{global_height} - creating single tile")
    else:
        rows, cols = calculate_grid(global_width, global_height, max_size)
        print(f"Image {global_width}x{global_height} -> slicing into {rows}x{cols} grid")
    
    # Calculate tile sizes (works for both 1x1 and NxM grids)
    base_tile_w = math.ceil(global_width / cols)
    base_tile_h = math.ceil(global_height / rows)
    overlap_w = int(base_tile_w * overlap) if rows > 1 or cols > 1 else 0
    overlap_h = int(base_tile_h * overlap) if rows > 1 or cols > 1 else 0
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile coordinates in global pixel space
            x1 = max(0, col * base_tile_w - overlap_w)
            y1 = max(0, row * base_tile_h - overlap_h)
            x2 = min(global_width, (col + 1) * base_tile_w + overlap_w)
            y2 = min(global_height, (row + 1) * base_tile_h + overlap_h)
            
            # Crop tile
            tile = img.crop((x1, y1, x2, y2))
            tile_w, tile_h = tile.size
            tile_id = f"tile_{row:02d}_{col:02d}"
            
            # Save tile temporarily for OCR
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tile_temp_path = tmp.name
                tile.save(tile_temp_path, "PNG")
            
            try:
                # Run OCR on tile
                if OCR_ENGINE == "paddleocr":
                    extractions, _, _ = extract_with_paddleocr(tile_temp_path)
                else:
                    extractions, _, _ = extract_with_easyocr(tile_temp_path)
                
                # Process extractions with BOTH coordinate systems
                tile_extractions = []
                for ext in extractions:
                    # Tile-relative coordinates (for vision tool)
                    bbox_tile_rel = normalize_to_tile_relative(
                        ext["bbox_pixels"], tile_w, tile_h
                    )
                    
                    # Global coordinates (for merge step)
                    bbox_global = normalize_to_global(
                        ext["bbox_pixels"], tile_w, tile_h,
                        x1, y1, global_width, global_height
                    )
                    
                    tile_ext = {
                        "text": ext["text"],
                        "bbox": bbox_tile_rel,  # TILE-RELATIVE for vision
                        "type": classify_text(ext["text"]),
                        "confidence": ext["confidence"]
                    }
                    tile_extractions.append(tile_ext)
                    
                    global_ext = {
                        "text": ext["text"],
                        "bbox": bbox_global,  # GLOBAL for merge
                        "bbox_global": bbox_global,
                        "type": classify_text(ext["text"]),
                        "confidence": ext["confidence"],
                        "source_tile": tile_id
                    }
                    all_global_extractions.append(global_ext)
                
                # Save per-tile OCR (with TILE-RELATIVE coordinates)
                tile_ocr_path = output_path / f"ocr_{tile_id}.json"
                tile_ocr_data = {
                    "tile_id": tile_id,
                    "coordinate_system": "tile_relative_0_1000",
                    "tile_size": {"width": tile_w, "height": tile_h},
                    "tile_position_global": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "total_detections": len(tile_extractions),
                    "extractions": tile_extractions
                }
                with open(tile_ocr_path, "w") as f:
                    json.dump(tile_ocr_data, f, indent=2)
                
                tile_outputs.append({
                    "tile_id": tile_id,
                    "ocr_file": str(tile_ocr_path),
                    "detections": len(tile_extractions)
                })
                print(f"  Tile [{row},{col}]: {len(tile_extractions)} detections -> {tile_ocr_path.name}")
            
            finally:
                # Windows may hold file locks - try cleanup with retry
                try:
                    os.unlink(tile_temp_path)
                except OSError:
                    # File may still be locked by PaddleOCR on Windows
                    import time
                    time.sleep(0.5)
                    try:
                        os.unlink(tile_temp_path)
                    except OSError:
                        pass  # Accept cleanup failure, temp dir will be cleaned eventually
    
    # Deduplicate global extractions (only matters when multiple tiles with overlap)
    if rows > 1 or cols > 1:
        all_global_extractions = deduplicate_ocr(all_global_extractions)
    
    # Group by type for summary
    grouped = {
        "equipment_tags": [e for e in all_global_extractions if e["type"] == "equipment_tag"],
        "instrument_tags": [e for e in all_global_extractions if e["type"] == "instrument_tag"],
        "valve_tags": [e for e in all_global_extractions if e["type"] == "valve_tag"],
        "line_numbers": [e for e in all_global_extractions if e["type"] == "line_number"],
        "measurements": [e for e in all_global_extractions if e["type"] == "measurement"],
        "labels": [e for e in all_global_extractions if e["type"] == "label"]
    }
    
    # Save global summary
    global_result = {
        "source_image": str(image_path),
        "image_size": {"width": global_width, "height": global_height},
        "ocr_engine": OCR_ENGINE.upper(),
        "coordinate_system": "global_0_1000",
        "total_detections": len(all_global_extractions),
        "tiles_processed": tile_outputs,
        "summary": {k: len(v) for k, v in grouped.items()},
        "detections": grouped,
        "all_extractions": all_global_extractions
    }
    
    global_output_path = output_path / "ocr_global.json"
    with open(global_output_path, "w") as f:
        json.dump(global_result, f, indent=2)
    print(f"\nGlobal summary: {global_output_path}")
    
    return global_result


def deduplicate_ocr(extractions: list, overlap_threshold: float = 0.7) -> list:
    """Remove duplicate detections from overlapping tiles."""
    if len(extractions) <= 1:
        return extractions
    
    sorted_ext = sorted(extractions, key=lambda x: x.get("confidence", 0), reverse=True)
    
    kept = []
    for ext in sorted_ext:
        is_dup = False
        for existing in kept:
            if ext["text"].upper() == existing["text"].upper():
                b1 = ext.get("bbox_global", ext["bbox"])
                b2 = existing.get("bbox_global", existing["bbox"])
                
                y_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                x_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                intersection = y_overlap * x_overlap
                
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                
                if area1 > 0 and intersection / area1 > overlap_threshold:
                    is_dup = True
                    break
        
        if not is_dup:
            kept.append(ext)
    
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="OCR pre-processing for P&ID with per-tile and global coordinate outputs"
    )
    parser.add_argument("image_path", help="Path to P&ID image")
    parser.add_argument(
        "--output-dir", "-o",
        default="./ocr_output",
        help="Output directory for OCR JSON files (default: ./ocr_output)"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=int,
        default=2048,
        help="Max tile size in pixels (default: 2048)"
    )
    parser.add_argument(
        "--overlap", "-v",
        type=float,
        default=0.1,
        help="Tile overlap percentage (default: 0.1)"
    )
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="Use high-recall mode with 50%% overlap (as in Digitize-PID paper)"
    )
    
    args = parser.parse_args()
    
    # High-recall mode overrides overlap to 50%
    overlap = 0.5 if args.high_recall else args.overlap
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    print(f"Processing: {args.image_path}")
    print(f"OCR Engine: {OCR_ENGINE or 'None available'}")
    print(f"Output dir: {args.output_dir}")
    if args.high_recall:
        print(f"Mode: HIGH-RECALL (50% overlap)")
    print()
    
    results = process_image(args.image_path, args.output_dir, args.max_size, overlap)
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES:")
    print(f"  Per-tile OCR (tile-relative coords): ocr_tile_XX_YY.json")
    print(f"  Global summary (global coords):      ocr_global.json")
    print(f"{'='*60}")
    print(f"Total detections: {results['total_detections']}")
    for k, v in results['summary'].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
