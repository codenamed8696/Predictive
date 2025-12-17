#!/usr/bin/env python3
"""
P&ID Extraction Pipeline

Single command wrapper that orchestrates the complete P&ID digitization workflow:
1. Check image size and slice if needed
2. Run OCR pre-processing
3. Generate extraction prompts for each tile
4. Merge results with validation

Usage:
    python pid_pipeline.py <image_path> --output result.json
    
Note: This script prepares the data for Claude/vision tool processing.
The actual vision extraction must be done via Claude Code with the MCP tool.

Install dependencies:
    pip install pillow paddleocr paddlepaddle
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Import sibling scripts
try:
    from slice_image import slice_image
    from ocr_preprocess import process_image as ocr_process
    from merge_json import merge_json_files
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from slice_image import slice_image
    from ocr_preprocess import process_image as ocr_process
    from merge_json import merge_json_files

try:
    from line_detect import detect_lines
except Exception:
    detect_lines = None

try:
    from shape_detect import detect_shapes
except Exception:
    detect_shapes = None


def create_extraction_prompts(tiles_dir: Path, ocr_dir: Path, output_dir: Path,
                               ocr_min_confidence: float = 0.7) -> list:
    """
    Generate extraction prompt files for each tile.
    OCR data is automatically embedded with confidence filtering.
    """
    prompts = []
    tiles_metadata_path = tiles_dir / "tiles_metadata.json"
    
    if tiles_metadata_path.exists():
        with open(tiles_metadata_path) as f:
            tiles_metadata = json.load(f)
        tiles = tiles_metadata.get("tiles", [])
    else:
        # Slicing now always occurs (1x1 for small images)
        # This fallback should not be reached, but kept for safety
        print("Warning: tiles_metadata.json not found - this should not happen")
        tiles = [{"filename": "tile_00_00.png", "tile_id": 0}]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for tile in tiles:
        tile_name = tile["filename"].replace(".png", "")
        ocr_file = ocr_dir / f"ocr_{tile_name}.json"
        
        # OCR now always uses tile naming (ocr_tile_00_00.json even for 1x1)
        
        # Load and filter OCR data by confidence
        ocr_data = []
        ocr_summary = {"equipment": [], "instruments": [], "valves": [], "lines": []}
        
        if ocr_file.exists():
            with open(ocr_file) as f:
                ocr_json = json.load(f)
                raw_ocr = ocr_json.get("extractions", [])
                
                # Filter by confidence and categorize
                for item in raw_ocr:
                    if item.get("confidence", 0) >= ocr_min_confidence:
                        ocr_data.append(item)
                        item_type = item.get("type", "label")
                        if item_type == "equipment_tag":
                            ocr_summary["equipment"].append(item["text"])
                        elif item_type == "instrument_tag":
                            ocr_summary["instruments"].append(item["text"])
                        elif item_type == "valve_tag":
                            ocr_summary["valves"].append(item["text"])
                        elif item_type == "line_number":
                            ocr_summary["lines"].append(item["text"])
        
        # Create enhanced prompt with OCR grounding
        ocr_json_str = json.dumps(ocr_data[:30], indent=2)
        
        # Determine extraction mode based on OCR availability
        has_ocr_data = len(ocr_data) > 0
        ocr_counts = {
            "equipment": len(ocr_summary["equipment"]),
            "instruments": len(ocr_summary["instruments"]),
            "valves": len(ocr_summary["valves"]),
            "lines": len(ocr_summary["lines"])
        }
        
        if has_ocr_data:
            # OCR-grounded extraction mode
            mode_instruction = f"""## OCR-GROUNDED EXTRACTION MODE

The following text was detected by OCR in this tile. Use these as the **source of truth** for tag names:

### Detected Equipment Tags: {ocr_summary['equipment'] or ['(none detected)']}
### Detected Instrument Tags: {ocr_summary['instruments'] or ['(none detected)']}
### Detected Valve Tags: {ocr_summary['valves'] or ['(none detected)']}
### Detected Line Numbers: {ocr_summary['lines'] or ['(none detected)']}

### Full OCR Data (with coordinates):
```json
{ocr_json_str}
```
{"... (showing first 30 of " + str(len(ocr_data)) + ")" if len(ocr_data) > 30 else ""}

## MINIMUM EXTRACTION REQUIREMENTS

Based on OCR detections, you MUST extract AT LEAST:
- Equipment: {ocr_counts['equipment']} (OCR found {ocr_counts['equipment']} equipment tags)
- Instruments: {ocr_counts['instruments']} (OCR found {ocr_counts['instruments']} instrument tags)  
- Valves: {ocr_counts['valves']} (OCR found {ocr_counts['valves']} valve tags)

If you extract FEWER entities than OCR detected, you MUST explain why in a "notes" field.

## CRITICAL ANTI-HALLUCINATION RULES

1. **ONLY use tags that appear in the OCR data above**
2. If a symbol is visible but its tag is NOT in OCR, use `"tag": null` and set `"tag_complete": false`
3. Never invent or guess tag names
4. Match OCR text to symbols using spatial proximity (OCR bbox near symbol bbox)"""
        else:
            # Exhaustive extraction mode (no OCR data available)
            mode_instruction = """## ⚠️ EXHAUSTIVE EXTRACTION MODE (NO OCR DATA)

**WARNING**: No OCR data is available for this tile. This significantly impacts extraction accuracy.

### EXHAUSTIVE EXTRACTION RULES

Since OCR grounding is unavailable, you must:
1. **Extract EVERY visible symbol** - err on the side of over-extraction
2. **Use approximate/partial tags** - transcribe any visible text on symbols even if incomplete
3. **Set lower confidence** - use `"confidence": 0.5` to `"confidence": 0.7` for all entities
4. **Flag uncertainty** - set `"tag_complete": false` for any tags you're unsure about
5. **Never skip symbols** - even if you can't read the tag, include the entity with `"tag": null`

### SYMBOL DETECTION PRIORITY

Scan the entire tile systematically:
1. Look for ALL circular symbols (instruments - ISA bubbles)
2. Look for ALL valve symbols (bowtie, gate, globe, control)
3. Look for ALL major equipment (vessels, tanks, pumps, exchangers)
4. Trace ALL piping lines and their connections

**It is better to extract 50 uncertain entities than miss 5 real ones.**"""
        
        prompt = {
            "tile_id": tile_name,
            "tile_path": str(tiles_dir / tile["filename"]),
            "ocr_data": ocr_data,
            "ocr_summary": ocr_summary,
            "has_ocr": has_ocr_data,
            "extraction_mode": "ocr_grounded" if has_ocr_data else "exhaustive",
            "extraction_prompt": f"""You are an expert Industrial Automation Engineer performing P&ID digitization.

{mode_instruction}

## YOUR TASK

Extract P&ID components from this tile:
1. **Equipment**: Vessels, tanks, pumps, heat exchangers (match to OCR equipment tags)
2. **Instruments**: ISA-5.1 bubbles (match to OCR instrument tags)
3. **Valves**: Manual/control valves (match to OCR valve tags)
4. **Lines**: Process piping with waypoints

## OUTPUT FORMAT

Return JSON:
```json
{{
  "extraction_mode": "{'ocr_grounded' if has_ocr_data else 'exhaustive'}",
  "nodes": {{
    "equipment": [{{"id": "EQ-001", "tag": "T-101", "class": "Vessel", "bbox": [y1,x1,y2,x2], "confidence": 0.9, "tag_complete": true}}],
    "instruments": [{{"id": "INST-001", "tag": "TIC-101", "function_letters": "TIC", "bbox": [...], "confidence": 0.9}}],
    "valves": [{{"id": "VLV-001", "tag": "V-101", "class": "Gate", "upstream": "EQ-001", "downstream": "EQ-002", "confidence": 0.9}}]
  }},
  "edges": {{
    "process_lines": [{{"id": "LINE-001", "source": {{"node_id": "EQ-001"}}, "target": {{"node_id": "VLV-001"}}, "route_waypoints": [[x,y], ...]}}]
  }},
  "notes": "Optional: explain any discrepancies between OCR count and extracted count"
}}
```

**Coordinates**: All bbox values use [ymin, xmin, ymax, xmax] in 0-1000 normalized space.
"""
        }
        
        prompt_path = output_dir / f"prompt_{tile_name}.json"
        with open(prompt_path, "w") as f:
            json.dump(prompt, f, indent=2)
        
        prompts.append(prompt_path)
    
    return prompts


def run_pipeline(
    image_path: str,
    output_dir: str = "./pid_output",
    max_tile_size: int = 2048,
    overlap: float = 0.1,
    skip_ocr: bool = False,
    skip_prompts: bool = False,
    skip_cv: bool = False,
    high_recall: bool = False
) -> dict:
    """
    Run the complete P&ID digitization pipeline.
    
    Returns:
        dict with paths to all generated artifacts
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "source_image": str(image_path),
        "timestamp": timestamp,
        "tiles_dir": None,
        "cv_dir": None,
        "ocr_dir": None,
        "prompts_dir": None,
        "tiles_metadata": None,
        "cv_lines": None,
        "cv_shapes": None,
        "ocr_global": None,
        "extraction_prompts": [],
    }
    
    print(f"\n{'='*60}")
    print(f"P&ID EXTRACTION PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    if high_recall:
        print(f"Mode: HIGH-RECALL (50% overlap)")
    print()
    
    # Determine effective overlap
    effective_overlap = 0.5 if high_recall else overlap
    
    # Step 1: Slice image
    print("[1/5] Slicing image into tiles...")
    tiles_dir = output_dir / "tiles"
    tiles_metadata = slice_image(str(image_path), str(tiles_dir), max_tile_size, effective_overlap)
    results["tiles_dir"] = str(tiles_dir)
    results["tiles_metadata"] = str(tiles_dir / "tiles_metadata.json")
    
    if tiles_metadata.get("total_tiles", 0) > 1:
        print(f"      -> Created {tiles_metadata['total_tiles']} tiles")
    else:
        print(f"      -> Created single tile (image within {max_tile_size}px)")
    print()
    
    # Step 2: CV Line Detection (Digitize-PID morphological approach)
    cv_dir = output_dir / "cv"
    if not skip_cv and detect_lines is not None:
        print("[2/5] Running CV line detection...")
        cv_dir.mkdir(parents=True, exist_ok=True)
        try:
            line_results = detect_lines(str(image_path), str(cv_dir))
            results["cv_dir"] = str(cv_dir)
            results["cv_lines"] = str(cv_dir / "cv_lines.json")
            print(f"      -> Detected {line_results['summary']['total_solid_lines']} solid lines")
            print(f"      -> Detected {line_results['summary']['total_dashed_lines']} dashed lines")
        except Exception as e:
            print(f"      -> CV line detection failed: {e}")
        print()
    else:
        if skip_cv:
            print("[2/5] Skipping CV line detection (--skip-cv)")
        else:
            print("[2/5] Skipping CV line detection (module not available)")
        print()
    
    # Step 3: CV Shape Detection (Hough circles for instruments)
    if not skip_cv and detect_shapes is not None:
        print("[3/5] Running CV shape detection...")
        try:
            shape_results = detect_shapes(str(image_path), str(cv_dir))
            results["cv_shapes"] = str(cv_dir / "cv_shapes.json")
            print(f"      -> Detected {shape_results['summary']['total_circles']} instrument candidates")
            print(f"      -> Detected {shape_results['summary']['total_rectangles']} rectangle candidates")
        except Exception as e:
            print(f"      -> CV shape detection failed: {e}")
        print()
    else:
        if skip_cv:
            print("[3/5] Skipping CV shape detection (--skip-cv)")
        else:
            print("[3/5] Skipping CV shape detection (module not available)")
        print()
    
    # Step 4: OCR pre-processing
    if not skip_ocr:
        print("[4/5] Running OCR pre-processing...")
        ocr_dir = output_dir / "ocr"
        ocr_results = ocr_process(str(image_path), str(ocr_dir), max_tile_size, effective_overlap)
        results["ocr_dir"] = str(ocr_dir)
        results["ocr_global"] = str(ocr_dir / "ocr_global.json")
        total_detections = ocr_results.get('total_detections', 0)
        print(f"      -> Detected {total_detections} text labels")
        
        # OCR validation gate: warn if no detections
        if total_detections == 0:
            print("      ⚠️  WARNING: OCR detected 0 text labels!")
            print("         This may indicate image quality issues or OCR failure.")
            print("         Extraction quality will be significantly reduced.")
        print()
    else:
        print("⚠️  [4/5] Skipping OCR (--force-no-ocr)")
        print("         EXTRACTION ACCURACY WILL BE SIGNIFICANTLY REDUCED")
        ocr_dir = output_dir / "ocr"
        print()
    
    # Step 5: Generate extraction prompts
    if not skip_prompts:
        print("[5/5] Generating extraction prompts...")
        prompts_dir = output_dir / "prompts"
        prompts = create_extraction_prompts(tiles_dir, ocr_dir, prompts_dir)
        results["prompts_dir"] = str(prompts_dir)
        results["extraction_prompts"] = [str(p) for p in prompts]
        print(f"      -> Created {len(prompts)} prompt files")
        print()
    else:
        print("[5/5] Skipping prompt generation (--skip-prompts)")
        print()
    
    # Save pipeline manifest
    manifest_path = output_dir / "pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Use Claude with MCP tool 'understand_technical_diagram' on each tile")
    print(f"  2. Save extraction results as tile_XX_XX.json in {output_dir}/extractions/")
    print(f"  3. Run: python merge_json.py {output_dir}/extractions/*.json -o final.json")
    print(f"  4. Visualize: python visualize_results.py {image_path} final.json")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="P&ID Digitization Pipeline - Prepares image for extraction"
    )
    parser.add_argument("image_path", help="Path to P&ID image")
    parser.add_argument(
        "--output-dir", "-o",
        default="./pid_output",
        help="Output directory (default: ./pid_output)"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=int, default=2048,
        help="Maximum tile size in pixels (default: 2048)"
    )
    parser.add_argument(
        "--overlap", "-v",
        type=float, default=0.1,
        help="Tile overlap percentage (default: 0.1)"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="DEPRECATED: Use --force-no-ocr instead. This flag still works but shows a warning."
    )
    parser.add_argument(
        "--force-no-ocr",
        action="store_true",
        help="Skip OCR pre-processing (NOT RECOMMENDED - significantly reduces extraction accuracy)"
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Skip prompt generation"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip CV line/shape detection"
    )
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="Use high-recall OCR mode with 50%% overlap"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Handle deprecated --skip-ocr flag
    skip_ocr_effective = args.skip_ocr or args.force_no_ocr
    if args.skip_ocr:
        print("⚠️  WARNING: --skip-ocr is DEPRECATED. Use --force-no-ocr instead.")
        print("   OCR grounding is essential for accurate extraction.")
        print()
    
    run_pipeline(
        args.image_path,
        args.output_dir,
        args.max_size,
        args.overlap,
        skip_ocr_effective,
        args.skip_prompts,
        args.skip_cv,
        args.high_recall
    )


if __name__ == "__main__":
    main()
