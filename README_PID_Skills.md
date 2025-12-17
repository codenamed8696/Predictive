# P&ID Digitization Skill - Developer Guide

A Claude Code skill for extracting structured data from P&ID (Piping & Instrumentation Diagram) images using vision LLMs and OCR preprocessing.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Workflow](#workflow)
- [Scripts Reference](#scripts-reference)
- [Schema Reference](#schema-reference)
- [Extending the Skill](#extending-the-skill)
- [Troubleshooting](#troubleshooting)

---

## Overview

This skill converts raster P&ID images into structured JSON data suitable for:
- Digital Twin construction
- Automated process simulation
- Equipment inventory management
- Control loop documentation

**Key Features:**
- Dynamic image slicing for high-resolution diagrams
- OCR pre-processing with PaddleOCR/EasyOCR for text grounding
- 7-phase extraction pipeline (Legend → Equipment → Instruments → Valves → Piping → Process Logic → Validation)
- ISA-5.1 standard compliance
- Normalized 0-1000 coordinate system

**Target Runtime:** GLM 4.6 via Claude Code with MCP vision tools (specifically `understand_technical_diagram`)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: P&ID Image                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Size Check                            │
│                    (> 2048px threshold)                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
┌─────────────────────────┐            ┌─────────────────────────┐
│   slice_image.py        │            │   Direct Processing     │
│   (Dynamic Grid)        │            │   (Single Image)        │
└─────────────────────────┘            └─────────────────────────┘
              │                                      │
              ▼                                      │
┌─────────────────────────┐                          │
│   ocr_preprocess.py     │◄─────────────────────────┤
│   (Per Tile or Full)    │                          │
└─────────────────────────┘                          │
              │                                      │
              ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MCP Vision Tool: understand_technical_diagram   │
│                  + OCR Grounding Data                            │
│                  (Phases 0-6 per tile)                           │
└─────────────────────────────────────────────────────────────────┘
              │                                      │
              ▼                                      │
┌─────────────────────────┐                          │
│   merge_json.py         │                          │
│   (Dedupe + Transform)  │                          │
└─────────────────────────┘                          │
              │                                      │
              └──────────────────┬───────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Consolidated JSON                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
pid-digitization/
├── SKILL.md                          # Main skill entry point (loaded by Claude)
├── README.md                         # This developer guide
├── scripts/
│   ├── pid_pipeline.py              # Single-command workflow orchestration
│   ├── slice_image.py               # Dynamic image tiling
│   ├── ocr_preprocess.py            # PaddleOCR/EasyOCR text extraction
│   ├── merge_json.py                # JSON deduplication, merge, and validation
│   ├── visualize_results.py         # Visual debugging (bbox overlay)
│   └── errors.py                    # Structured error handling module
└── references/
    ├── extraction-guide.md          # Phase-by-phase extraction instructions
    ├── complete-schema.md           # Full JSON output schema
    └── detection-rules.md           # ISA-5.1 symbol detection tables
```

---

## Installation

### Prerequisites

```bash
# Required
pip install pillow

# For OCR (choose one)
pip install paddleocr paddlepaddle  # Recommended - best accuracy
# OR
pip install easyocr                  # Alternative - simpler setup
```

### Claude Code Plugin Registration

The skill is registered in `.claude-plugin/marketplace.json` under `industrial-skills`:

```bash
/plugin install industrial-skills@anthropic-agent-skills
```

---

## Workflow

### Step 1: Image Size Check

```python
# Threshold: 2048px in either dimension
if width > 2048 or height > 2048:
    # Slice image
else:
    # Process directly
```

### Step 2: Slice Image (if needed)

```bash
python scripts/slice_image.py input.png --output-dir ./tiles --max-size 2048 --overlap 0.1
```

**Output:** `tiles/` directory containing:
- `tile_00_00.png`, `tile_00_01.png`, ... (tile images)
- `tiles_metadata.json` (coordinates for each tile)

### Step 3: OCR Pre-Processing

```bash
python scripts/ocr_preprocess.py input.png --output-dir ./ocr_output --max-size 2048
```

**Output:** `ocr_output/` directory containing:
- `ocr_tile_00_00.json` - Tile-relative 0-1000 coords (use with tile_00_00.png)
- `ocr_tile_00_01.json` - etc.
- `ocr_global.json` - Global 0-1000 coords (for merge step)

**Per-tile OCR structure:**
```json
{
  "tile_id": "tile_00_00",
  "coordinate_system": "tile_relative_0_1000",
  "extractions": [
    {
      "text": "TK-228A",
      "bbox": [150, 200, 170, 280],
      "type": "equipment_tag",
      "confidence": 0.95
    }
  ]
}
```

> ⚠️ **CRITICAL**: Use per-tile OCR files when processing tiles!
> Global coordinates will NOT match what the vision tool sees.

### Step 4: Vision Extraction with OCR Grounding

For **each tile**, use the **matching per-tile OCR file**:

```
For tile_00_00.png → use ocr_tile_00_00.json
For tile_00_01.png → use ocr_tile_00_01.json
```

Prompt template:
```
I am providing a P&ID TILE and OCR data for THIS TILE.
Coordinates are 0-1000 scale RELATIVE TO THIS TILE.

**OCR DATA:**
```json
[paste "extractions" from ocr_tile_XX_YY.json]
```
**TASK:** Extract Phase 1 Equipment...
```

### Step 5: Merge Results

```bash
python scripts/merge_json.py ./tiles/*.json --tiles-metadata ./tiles/tiles_metadata.json --output final.json
```


---

## Scripts Reference

### pid_pipeline.py ⭐ NEW

**Purpose:** Single-command workflow that orchestrates slicing, OCR, and prompt generation.

```bash
python pid_pipeline.py <image_path> [options]

Options:
  --output-dir, -o    Output directory (default: ./pid_output)
  --max-size, -m      Max tile size in pixels (default: 2048)
  --overlap, -v       Tile overlap percentage (default: 0.1)
  --skip-ocr          Skip OCR pre-processing
  --skip-prompts      Skip prompt generation
```

**Output Structure:**
```
pid_output/
├── tiles/                  # Sliced images + tiles_metadata.json
├── ocr/                    # Per-tile and global OCR results
├── prompts/                # Extraction prompts for each tile
└── pipeline_manifest.json  # Paths to all artifacts
```

---

### slice_image.py

**Purpose:** Split high-resolution images into processable tiles.

```bash
python slice_image.py <image_path> [options]

Options:
  --output-dir, -o    Output directory (default: ./tiles)
  --max-size, -m      Max tile dimension in pixels (default: 2048)
  --overlap, -v       Overlap percentage 0.0-0.5 (default: 0.1)
```

**Algorithm:**
1. Calculate grid dimensions based on image size
2. Apply overlap to catch boundary elements
3. Save tiles with normalized coordinate metadata

---

### ocr_preprocess.py

**Purpose:** Extract and classify text labels with per-tile and global coordinates.

```bash
python ocr_preprocess.py <image_path> [options]

Options:
  --output-dir, -o    Output directory (default: ./ocr_output)
  --max-size, -m      Max tile size for slicing (default: 2048)
  --overlap, -v       Tile overlap percentage (default: 0.1)
```

**Dual Output Mode:**
1. **Per-tile OCR** (`ocr_tile_XX_YY.json`): Tile-relative 0-1000 coords for vision tool
2. **Global OCR** (`ocr_global.json`): Global 0-1000 coords for merge step

**Text Classification:**
- `equipment_tag`: T-101, P-102A, TK-100
- `instrument_tag`: TIC-101, PT-200, FV-300
- `valve_tag`: V212, HV-100
- `line_number`: 4"-PA-001
- `measurement`: 150 PSI, 200°C
- `label`: Everything else

---

### merge_json.py

**Purpose:** Combine tile extractions into unified output with validation.

```bash
python merge_json.py <json_files...> [options]

Options:
  --tiles-metadata, -t    Path to tiles_metadata.json
  --output, -o            Output file (default: merged_output.json)
  --validate, -v          Print detailed validation warnings
```

**Features:**
1. **Deduplication**: Match by tag (exact) or spatial overlap (>80%)
2. **Coordinate Transform**: Tile-local to global 0-1000 space
3. **Auto ID Generation**: Assigns `EQ-XXX`, `INST-XXX`, `VLV-XXX` to entities missing IDs
4. **Reference Validation**: Checks all cross-references resolve correctly
5. **Orphan Detection**: Flags equipment with no connections

**Validation Output:**
```
Merged 4 files -> output.json
  Equipment: 7
  Instruments: 8
  Valves: 26
  Lines: 14
  Validation: ✓ PASSED
```

---

### visualize_results.py ⭐ NEW

**Purpose:** Visual debugging - overlays extraction bboxes on the original image.

```bash
python visualize_results.py <image_path> <json_path> [options]

Options:
  --output, -o          Output image path (else displays in viewer)
  --min-confidence, -c  Filter by minimum confidence (0.0-1.0)
  --no-equipment        Hide equipment boxes
  --no-instruments      Hide instrument boxes
  --no-valves           Hide valve boxes
  --no-lines            Hide line waypoints
  --show-ocr            Show OCR text extractions
```

**Color Coding:**
| Entity Type | Color |
|-------------|-------|
| Equipment | Blue |
| Instruments | Green |
| Valves | Red |
| Lines | Yellow |
| OCR Text | Cyan |

---

## Schema Reference

See `references/complete-schema.md` for full schema. Key structures:

### Equipment Entity
```json
{
  "id": "EQ-001",
  "tag": "TK-228A",
  "class": "Tank",
  "bbox": [ymin, xmin, ymax, xmax],
  "anchors": {"inlet": {"x": 500, "y": 100}},
  "connections": {
    "inlet_from": ["valve_id"],
    "outlet_to": ["equipment_id"]
  },
  "interlocks": ["interlock_id"],
  "confidence": 0.85
}
```

### Process Logic
```json
{
  "flow_sequences": [...],
  "control_relationships": [
    {
      "sensor": "LT-9616",
      "controller": "LC-2727",
      "final_element": "LV-2727",
      "control_action": "Reverse"
    }
  ],
  "interlocks": [...]
}
```

---

## Extending the Skill

### Adding New Equipment Classes

Edit `references/detection-rules.md`:

```markdown
| Equipment Class | Visual Signature | Tag Pattern |
|----------------|------------------|-------------|
| NewType        | Description      | XX-NNN      |
```

### Adding New OCR Patterns

Edit `scripts/ocr_preprocess.py`, function `classify_text()`:

```python
# Add new pattern
if re.match(r'^NEW-PATTERN$', text):
    return "new_type_tag"
```

### Modifying Extraction Phases

Edit `references/extraction-guide.md` to add/modify phase instructions.

### Changing Coordinate System

Currently uses 0-1000 normalized scale. To change:
1. Update `normalize_bbox()` in `ocr_preprocess.py`
2. Update `transform_coordinates()` in `merge_json.py`
3. Update Core Invariants in `SKILL.md`

---

## Troubleshooting

### Poor Text Recognition

**Symptom:** Tags not detected or misread.

**Solutions:**
1. Ensure PaddleOCR installed (better than EasyOCR for technical text)
2. Check image resolution (increase DPI if possible)
3. Pre-process image (increase contrast, remove noise)

### Coordinate Mismatch

**Symptom:** OCR bbox doesn't match vision output locations.

**Solutions:**
1. Verify both use 0-1000 normalized scale
2. Check `tiles_metadata.json` tile offsets
3. Ensure same overlap percentage for OCR and slicing

### Memory Issues with Large Images

**Symptom:** Out of memory errors.

**Solutions:**
1. Reduce `--max-size` (e.g., 1024 instead of 2048)
2. Process tiles sequentially instead of parallel
3. Use `del` to free memory after processing each tile

### Vision LLM Hallucinating Tags

**Symptom:** LLM invents equipment that doesn't exist.

**Solutions:**
1. Always use OCR grounding - include OCR data in prompt
2. Use explicit instruction: "ONLY report equipment found in OCR data"
3. Cross-validate against OCR `all_extractions` in post-processing

---

## Contributing

When modifying this skill:

1. **SKILL.md** - Keep concise, actionable instructions only
2. **References** - Put detailed schemas/tables here, not in SKILL.md
3. **Scripts** - Test with actual P&ID images before committing
4. **Coordinate System** - Always use 0-1000 normalized scale

---

*Version: 1.0.0 | ISA-5.1 Compliant | GLM 4.6 + MCP Vision Tools*
