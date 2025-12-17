# Changelog

All notable changes to the P&ID Digitization Skill.

## [1.3.0] - 2024-12-17

### Added (Research Paper Adoption - arXiv:2412.12898, arXiv:2411.13929)
- **Agentic Critic + Repair Loop** (`critic.py`): Classify errors, apply repairs, revalidate
  - Error types: DANGLING_REF, DUPLICATE_ID, SEAM_BREAK, UNGROUNDED_TAG, ORPHAN_NODE
  - Reliability gates: graph soundness, seam integrity, evidence gate, provenance
- **Edge-Centric Metrics**: `evaluate_pipeline.py` now tracks:
  - `edge_connection_rate`, `dangling_ref_rate`, `seam_break_rate`
  - `tag_grounding_rate` for OCR evidence validation
- **Provenance Tracking**: All nodes/edges get `_source` field for auditability

---

## [1.2.0] - 2024-12-16

### Added (Digitize-PID Paper Adoption)
- **CV Line Detection** (`line_detect.py`): Morphological line detection with horizontal/vertical kernels, skeletonization, and DBSCAN-based dashed-line clustering for signal lines
- **CV Shape Detection** (`shape_detect.py`): Hough circle detection for instrument bubbles, rectangle detection for panels
- **BFS Label Propagation**: `merge_json.py` now propagates line labels along connected segments using BFS graph traversal
- **High-Recall OCR Mode**: `--high-recall` flag for 50% tile overlap (as recommended in Digitize-PID paper)
- **Evaluation Framework** (`evaluate_pipeline.py`): A/B/C pipeline comparison with metrics for OCR validation rate, label coverage, unresolved refs, orphans

### Changed
- **Pipeline is now 5 steps**: Slice -> CV Lines -> CV Shapes -> OCR -> Prompts
- **New CLI flags**: `--skip-cv`, `--high-recall` for pipeline control

---

## [1.1.2] - 2024-12-16

### Fixed (Developer Review Findings)
- **Mandatory Slicing Enforced in Code**: `slice_image.py` and `ocr_preprocess.py` now always create tiles (1x1 for small images) - no more early returns
- **Schema Version Alignment**: All files now use version `1.1.1` consistently (was 1.0.0 in merge output, 1.1.0 in schema doc)
- **Legend line_types Merging**: `merge_json.py` now properly merges `legend.line_types` arrays (was only merging symbols/abbreviations)
- **OCR Fuzzy Matching**: `fuzzy_threshold` parameter is now actually used - implements similarity ratio matching
- **Pipeline Path Bug**: Fixed `pid_pipeline.py` fallback from non-existent `full_image.png` to proper `tile_00_00.png`
- **Console Encoding**: Replaced Unicode arrows with ASCII equivalents for Windows compatibility

---

## [1.1.1] - 2024-12-15

### Changed
- **Mandatory Slicing**: Image slicing is now **required for ALL images** regardless of size
  - Ensures consistent coordinate handling and OCR-to-vision alignment
  - Removed conditional 2048px threshold logic
  - Workflow simplified to 4 steps: Slice -> OCR -> Extract -> Merge

---

## [1.1.0] - 2024-12-14

### Added
- **Signal Lines Transform**: `merge_json.py` now transforms `signal_lines` waypoints from tile-local to global coordinates (same as `process_lines`)
- **Geometry-Based Line Deduplication**: New `deduplicate_lines_by_geometry()` function uses waypoint path similarity instead of tag/ID matching
- **Waypoint Similarity Scoring**: `calculate_waypoint_similarity()` computes path similarity for robust duplicate detection
- **Orthogonality Validation**: `check_waypoint_orthogonality()` detects non-horizontal/vertical pipe segments with 5° tolerance
- **Structured Validation Output**: `validate_references()` now returns structured dict with:
  - `unresolved_references[]` - array with `{source_phase, source_id, missing_reference, field}`
  - `incomplete_loops[]` - control loops missing sensor or final element
  - `orthogonality_warnings[]` - diagonal segment detections
- **Control Loop Completeness Check**: Validates that each control relationship has both sensor and final_element
- **Extended Legend Format**: `line_types` schema now includes `color`, `size`, `description` fields
- **Notes Field**: Metadata schema now includes `notes[]` array
- **PaddleX OCR Compatibility**: `ocr_preprocess.py` now supports new PaddleX inference result format (`rec_texts`, `rec_scores`, `dt_polys` keys)

### Changed
- **Schema Version**: Bumped from 1.0.0 to 1.1.0
- **Sample JSON**: `pid_digitization_complete_result.json` rewritten with:
  - Correct `[ymin, xmin, ymax, xmax]` bbox format
  - Canonical field names (`function_letters` instead of `isa_code`, `class` instead of `type`)
  - Structured source/target objects in process_lines
  - Proper entity ID format (EQ-XXX, INST-XXX, VLV-XXX)
- **Validation Section**: Now includes `unresolved_references`, `incomplete_loops`, `orthogonality_warnings`
- **Total Lines Count**: Now includes both `process_lines` and `signal_lines`
- **OffPage Handling**: Line source/target with `type: "OffPage"` excluded from reference validation

### Fixed
- Signal lines waypoints were not being transformed to global coordinates
- Lines with missing/null IDs were incorrectly grouped during deduplication
- Reference validation falsely flagged OffPage connections as unresolved

### Deprecated
- None

### Removed
- None

### Developer Notes
- `validate_references()` return type changed from `list` to `dict`
- New functions added: `calculate_waypoint_similarity()`, `check_waypoint_orthogonality()`, `deduplicate_lines_by_geometry()`
- `all_signal_lines` collection added to merge pipeline

---

## [1.0.0] - 2024-12-01

### Added
- Initial release with 7-phase extraction pipeline
- OCR pre-processing with PaddleOCR/EasyOCR support
- Dynamic image tiling for high-resolution diagrams
- Tile-relative and global coordinate systems
- ISA-5.1 compliant instrument/valve detection
- OCR-grounded extraction prompts with anti-hallucination measures
- Merge pipeline with tag-based deduplication
- Visual debugging overlay tool
