---
name: pid-digitization
description: Extract structured data from P&ID (Piping & Instrumentation Diagram) images. Use when users provide P&ID diagrams and need to digitize equipment, instruments, valves, piping topology, and process logic into JSON for Digital Twin construction, simulation, or documentation. Triggers include requests to analyze P&IDs, extract equipment lists, identify control loops, trace piping routes, or convert engineering drawings to structured data.
---

# P&ID Digitization Skill

Extract structured JSON data from P&ID raster images using MCP vision tools and auxiliary processing scripts.

---

## MANDATORY: MCP Tool Selection

> [!CAUTION]
> Do NOT use generic `analyze_image` or `extract_text_from_screenshot` tools.
> You MUST use the specialized technical diagram tool.

**Required MCP Tool**: `understand_technical_diagram`

```
Tool: zai-mcp-server - understand_technical_diagram
Parameters:
  - image_source: <path to P&ID image or tile>
  - diagram_type: "piping-instrumentation"
  - prompt: <phase-specific extraction prompt from extraction-guide.md>
```

---

## MANDATORY: Pre-Processing - Always Slice

> [!IMPORTANT]
> Before ANY extraction, you MUST slice the image into tiles. This is **mandatory for ALL image sizes** to ensure consistent coordinate handling and OCR alignment.

### Step 1: Slice the Image

```bash
# REQUIRED for ALL images - no exceptions
python scripts/slice_image.py <image_path> --output-dir ./tiles --max-size 2048 --overlap 0.1
```

- This creates tiles in `./tiles/` with `tiles_metadata.json`
- **MUST** process EACH tile separately (Step 2)
- **MUST** merge results afterward (Step 3)

> [!NOTE]
> Even small images benefit from the slicing pipeline as it ensures consistent coordinate transformation and proper OCR-to-vision alignment.

### Step 2: OCR Pre-Processing 

Run OCR to extract text labels. The script outputs **per-tile OCR files** with **tile-relative coordinates**:
```bash
python scripts/ocr_preprocess.py <image_path> --output-dir ./ocr_output --max-size 2048
```

**Output Files**:
- `ocr_output/ocr_tile_00_00.json` - Tile-relative 0-1000 coords (for vision tool)
- `ocr_output/ocr_tile_00_01.json` - etc.
- `ocr_output/ocr_global.json` - Global 0-1000 coords (for merge step)

> [!CAUTION]
> When processing a tile, use ONLY the matching `ocr_tile_XX_YY.json` file.
> Do NOT use global coordinates - they won't match the vision tool's view.

**OCR-Grounded Vision Prompt Template** (for each tile):
```
I am providing you with a P&ID diagram TILE and OCR text labels for THIS TILE.
Coordinates are [ymin, xmin, ymax, xmax] in 0-1000 scale RELATIVE TO THIS TILE.

**OCR DATA FOR THIS TILE:**
```json
[PASTE "extractions" from ocr_tile_XX_YY.json]
```

**CRUCIAL INSTRUCTION**: 
- Use OCR Data to identify tag names when image text is unclear
- The bbox coordinates match this tile's coordinate space

**YOUR TASK**: [Phase-specific extraction task]
```

### Step 3: Extract with MCP Vision Tool

For **each tile**, call `understand_technical_diagram`:

```
understand_technical_diagram(
  image_source: "<tile_path>",
  diagram_type: "piping-instrumentation",
  prompt: "Extract Phase 1 Equipment: Identify all equipment symbols, extract tags 
           (format: X-NNN), classify type (Vessel/Pump/Tank/etc), locate bounding 
           boxes in normalized 0-1000 coordinates. Return JSON per complete-schema.md."
)
```

Run these phases in order:
1. **Phase 0**: Legend & Metadata
2. **Phase 1**: Equipment (tags, class, anchors, connections)
3. **Phase 2**: Instruments (ISA codes, control loops)
4. **Phase 3**: Valves (type, actuator, inline position)
5. **Phase 4**: Piping (source, target, waypoints, inline components)
6. **Phase 5**: Process Logic (flow sequences, control relationships, interlocks)
7. **Phase 6**: Validation (cross-references, orphan detection)

Save each tile's output as JSON: `tile_00_00.json`, `tile_00_01.json`, etc.

### Step 4: Merge Results (REQUIRED)

```bash
python scripts/merge_json.py ./tiles/*.json --tiles-metadata ./tiles/tiles_metadata.json --output final_output.json
```

### Step 5: Critic + Repair (RECOMMENDED)

Run the agentic critic to validate and repair the merged output:

```bash
python scripts/critic.py final_output.json --ocr-json ./ocr_output/ocr_global.json -out repaired.json -r critic_report.json
```

The critic detects and repairs:
- **DANGLING_REF**: Edges referencing missing nodes
- **DUPLICATE_ID**: Duplicate entity IDs
- **SEAM_BREAK**: Lines broken at tile boundaries
- **UNGROUNDED_TAG**: Tags not found in OCR data

---

## Core Invariants

### Coordinate System
- **Normalized space**: Top-Left = `[0, 0]`, Bottom-Right = `[1000, 1000]`
- All items return `bbox`: `[ymin, xmin, ymax, xmax]`
- OCR and Vision use same coordinate system for matching

> [!CAUTION]
> NEVER use raw pixel coordinates in output. ALL coordinates must be normalized to 0-1000 scale.

### Entity ID Format
- Equipment: `EQ-XXX` (e.g., `EQ-001`)
- Instruments: `INST-XXX` (e.g., `INST-001`)
- Valves: `VLV-XXX` (e.g., `VLV-001`)
- Lines: `LINE-XXX` (auto-generated in merge step if missing)

### Text Association Rules
| Rule | Description |
|------|-------------|
| Proximity | Associate labels within 50-unit radius of symbol |
| Enclosure | Text inside symbol boundary belongs to that symbol |
| Line-Parallel | Text parallel to pipe describes that line |

### Orthogonality
- P&ID pipes NEVER travel diagonally
- Use `waypoints` array for corners: `[[x1,y1], [x2,y2], ...]`

---

## Output Rules

- Return **valid JSON only**—no markdown commentary
- Use `null` for unknown values (never fabricate)
- Include `confidence` scores (0.0–1.0) where specified
- Embed connection/interlock data within equipment entities
- Trust OCR for tag text, Vision for symbol type and topology

---

## Reference Files

| File | Contents |
|------|----------|
| [extraction-guide.md](./references/extraction-guide.md) | **Phase-by-phase extraction instructions** |
| [complete-schema.md](./references/complete-schema.md) | Full JSON schema for all phases |
| [detection-rules.md](./references/detection-rules.md) | Equipment, instrument, valve detection tables |

---

## Error Handling

- Unreadable region: `{"error": "unreadable", "region_bbox": [y1,x1,y2,x2]}`
- Ambiguous symbol: Include with `"confidence": 0.3-0.5` and `"ambiguity_note"`
- Partial tag: `"tag": "T-10?", "tag_complete": false`
- OCR/Vision mismatch: Trust OCR for text, Vision for symbol type

---

> **Schema Version**: 1.3.0 | See [CHANGELOG.md](./CHANGELOG.md) for version history
