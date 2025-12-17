#!/usr/bin/env python3
"""
P&ID JSON Merge Utility

Merges multiple tile extraction JSONs into a single consolidated output.
Handles deduplication, coordinate transformation, and boundary reconnection.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# ID counters for unique ID generation
_id_counters = {
    "equipment": 0,
    "instrument": 0,
    "valve": 0,
    "line": 0,
    "signal_line": 0,
    "interlock": 0
}


def generate_unique_id(entity_type: str) -> str:
    """Generate a unique ID for an entity type."""
    global _id_counters
    prefixes = {
        "equipment": "EQ",
        "instrument": "INST",
        "valve": "VLV",
        "line": "LINE",
        "signal_line": "SIG",
        "interlock": "IL"
    }
    prefix = prefixes.get(entity_type, "ID")
    _id_counters[entity_type] = _id_counters.get(entity_type, 0) + 1
    return f"{prefix}-{_id_counters[entity_type]:03d}"


def reset_id_counters():
    """Reset all ID counters (for testing)."""
    global _id_counters
    for key in _id_counters:
        _id_counters[key] = 0


def validate_references(merged: dict) -> dict:
    """
    Validate all cross-references in the merged data.
    
    Returns:
        dict with:
        - unresolved_references: structured array with source_phase, source_id, missing_reference, field
        - warnings: string array for human-readable warnings
        - incomplete_loops: list of control loops missing sensor or final element
    """
    unresolved = []
    warnings = []
    incomplete_loops = []
    
    # Collect all valid IDs
    valid_ids = set()
    for eq in merged.get("nodes", {}).get("equipment", []):
        valid_ids.add(eq.get("id"))
        valid_ids.add(eq.get("tag"))  # Tags are also valid references
    for inst in merged.get("nodes", {}).get("instruments", []):
        valid_ids.add(inst.get("id"))
        valid_ids.add(inst.get("tag"))
    for valve in merged.get("nodes", {}).get("valves", []):
        valid_ids.add(valve.get("id"))
        valid_ids.add(valve.get("tag"))
    for line in merged.get("edges", {}).get("process_lines", []):
        valid_ids.add(line.get("id"))
    for line in merged.get("edges", {}).get("signal_lines", []):
        valid_ids.add(line.get("id"))
    for util in merged.get("utilities", []):
        valid_ids.add(util.get("id"))
    for loop in merged.get("control_loops", []):
        valid_ids.add(loop.get("loop_id"))

    for seq in merged.get("process_logic", {}).get("flow_sequences", []):
        valid_ids.add(seq.get("sequence_id"))
    for interlock in merged.get("process_logic", {}).get("interlocks", []):
        valid_ids.add(interlock.get("interlock_id"))
    
    valid_ids.discard(None)
    
    # Check equipment connections
    for eq in merged.get("nodes", {}).get("equipment", []):
        connections = eq.get("connections", {})
        for ref in connections.get("inlet_from", []):
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 1,
                    "source_id": eq.get("id"),
                    "missing_reference": ref,
                    "field": "connections.inlet_from"
                })
                warnings.append(f"Equipment {eq.get('id')}: inlet_from references unknown '{ref}'")
        for ref in connections.get("outlet_to", []):
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 1,
                    "source_id": eq.get("id"),
                    "missing_reference": ref,
                    "field": "connections.outlet_to"
                })
                warnings.append(f"Equipment {eq.get('id')}: outlet_to references unknown '{ref}'")

        for ref in eq.get("interlocks", []) or []:
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 1,
                    "source_id": eq.get("id"),
                    "missing_reference": ref,
                    "field": "interlocks"
                })
                warnings.append(f"Equipment {eq.get('id')}: interlocks references unknown '{ref}'")
    
    # Check valve upstream/downstream
    for valve in merged.get("nodes", {}).get("valves", []):
        if valve.get("upstream") and valve.get("upstream") not in valid_ids:
            unresolved.append({
                "source_phase": 3,
                "source_id": valve.get("id"),
                "missing_reference": valve.get("upstream"),
                "field": "upstream"
            })
            warnings.append(f"Valve {valve.get('id')}: upstream references unknown '{valve.get('upstream')}'")
        if valve.get("downstream") and valve.get("downstream") not in valid_ids:
            unresolved.append({
                "source_phase": 3,
                "source_id": valve.get("id"),
                "missing_reference": valve.get("downstream"),
                "field": "downstream"
            })
            warnings.append(f"Valve {valve.get('id')}: downstream references unknown '{valve.get('downstream')}'")
        if valve.get("controlled_by") and valve.get("controlled_by") not in valid_ids:
            unresolved.append({
                "source_phase": 3,
                "source_id": valve.get("id"),
                "missing_reference": valve.get("controlled_by"),
                "field": "controlled_by"
            })
            warnings.append(f"Valve {valve.get('id')}: controlled_by references unknown '{valve.get('controlled_by')}'")
    
    # Check line sources/targets
    for line in merged.get("edges", {}).get("process_lines", []):
        source = line.get("source", {})
        target = line.get("target", {})
        # Skip OffPage types (they won't have valid node_ids)
        if source.get("node_id") and source.get("type") != "OffPage" and source.get("node_id") not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": line.get("id"),
                "missing_reference": source.get("node_id"),
                "field": "source.node_id"
            })
            warnings.append(f"Line {line.get('id')}: source references unknown '{source.get('node_id')}'")
        if target.get("node_id") and target.get("type") != "OffPage" and target.get("node_id") not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": line.get("id"),
                "missing_reference": target.get("node_id"),
                "field": "target.node_id"
            })
            warnings.append(f"Line {line.get('id')}: target references unknown '{target.get('node_id')}'")

        for ref in line.get("inline_components", []) or []:
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 4,
                    "source_id": line.get("id"),
                    "missing_reference": ref,
                    "field": "inline_components"
                })
                warnings.append(f"Line {line.get('id')}: inline_components references unknown '{ref}'")

    # Check signal line sources/targets
    for line in merged.get("edges", {}).get("signal_lines", []):
        source = line.get("source", {})
        target = line.get("target", {})

        source_id = source.get("instrument_id")
        target_id = target.get("instrument_id")

        if source_id and source_id not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": line.get("id"),
                "missing_reference": source_id,
                "field": "source.instrument_id"
            })
            warnings.append(f"Signal line {line.get('id')}: source references unknown '{source_id}'")

        if target_id and target_id not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": line.get("id"),
                "missing_reference": target_id,
                "field": "target.instrument_id"
            })
            warnings.append(f"Signal line {line.get('id')}: target references unknown '{target_id}'")

        loop_id = line.get("loop_id")
        if loop_id and loop_id not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": line.get("id"),
                "missing_reference": loop_id,
                "field": "loop_id"
            })
            warnings.append(f"Signal line {line.get('id')}: loop_id references unknown '{loop_id}'")

    # Check instrument connected_to
    for inst in merged.get("nodes", {}).get("instruments", []):
        connected_to = inst.get("connected_to")
        if connected_to and connected_to not in valid_ids:
            unresolved.append({
                "source_phase": 2,
                "source_id": inst.get("id"),
                "missing_reference": connected_to,
                "field": "connected_to"
            })
            warnings.append(f"Instrument {inst.get('id')}: connected_to references unknown '{connected_to}'")

    # Check utilities connected_to
    for util in merged.get("utilities", []):
        connected_to = util.get("connected_to")
        if connected_to and connected_to not in valid_ids:
            unresolved.append({
                "source_phase": 4,
                "source_id": util.get("id"),
                "missing_reference": connected_to,
                "field": "connected_to"
            })
            warnings.append(f"Utility {util.get('id')}: connected_to references unknown '{connected_to}'")

    # Check control loop instrument references
    for loop in merged.get("control_loops", []):
        loop_id = loop.get("loop_id")
        for inst_id in loop.get("instruments", []) or []:
            if inst_id and inst_id not in valid_ids:
                unresolved.append({
                    "source_phase": 2,
                    "source_id": loop_id,
                    "missing_reference": inst_id,
                    "field": "control_loops.instruments"
                })
                warnings.append(f"Control loop {loop_id}: instruments references unknown '{inst_id}'")
    
    # Check control loop completeness
    for rel in merged.get("process_logic", {}).get("control_relationships", []):
        loop_id = rel.get("loop_id")
        has_sensor = rel.get("sensor") is not None
        has_final = rel.get("final_element") is not None

        for field in ["sensor", "controller", "final_element"]:
            ref = rel.get(field)
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 5,
                    "source_id": loop_id,
                    "missing_reference": ref,
                    "field": f"control_relationships.{field}"
                })
                warnings.append(f"Control relationship {loop_id}: {field} references unknown '{ref}'")
        
        if not has_sensor or not has_final:
            incomplete_loops.append(loop_id)
            if not has_sensor:
                warnings.append(f"Control loop {loop_id}: missing sensor")
            if not has_final:
                warnings.append(f"Control loop {loop_id}: missing final element")

    for seq in merged.get("process_logic", {}).get("flow_sequences", []):
        seq_id = seq.get("sequence_id")
        for node_id in seq.get("nodes", []) or []:
            if node_id and node_id not in valid_ids:
                unresolved.append({
                    "source_phase": 5,
                    "source_id": seq_id,
                    "missing_reference": node_id,
                    "field": "flow_sequences.nodes"
                })
                warnings.append(f"Flow sequence {seq_id}: nodes references unknown '{node_id}'")

    for interlock in merged.get("process_logic", {}).get("interlocks", []):
        interlock_id = interlock.get("interlock_id")
        trigger_instrument = interlock.get("trigger_instrument")
        if trigger_instrument and trigger_instrument not in valid_ids:
            unresolved.append({
                "source_phase": 5,
                "source_id": interlock_id,
                "missing_reference": trigger_instrument,
                "field": "interlocks.trigger_instrument"
            })
            warnings.append(f"Interlock {interlock_id}: trigger_instrument references unknown '{trigger_instrument}'")

        for ref in interlock.get("affected_equipment", []) or []:
            if ref and ref not in valid_ids:
                unresolved.append({
                    "source_phase": 5,
                    "source_id": interlock_id,
                    "missing_reference": ref,
                    "field": "interlocks.affected_equipment"
                })
                warnings.append(f"Interlock {interlock_id}: affected_equipment references unknown '{ref}'")
    
    return {
        "unresolved_references": unresolved,
        "warnings": warnings,
        "incomplete_loops": incomplete_loops
    }


def detect_orphans(merged: dict) -> list:
    """
    Detect equipment with no connections.
    Returns list of orphaned node IDs.
    """
    orphans = []
    
    # Collect all referenced equipment
    referenced = set()
    
    # From connections
    for eq in merged.get("nodes", {}).get("equipment", []):
        connections = eq.get("connections", {})
        for ref in connections.get("inlet_from", []):
            referenced.add(ref)
        for ref in connections.get("outlet_to", []):
            referenced.add(ref)
    
    # From valve upstream/downstream
    for valve in merged.get("nodes", {}).get("valves", []):
        referenced.add(valve.get("upstream"))
        referenced.add(valve.get("downstream"))
    
    # From lines
    for line in merged.get("edges", {}).get("process_lines", []):
        source = line.get("source", {})
        target = line.get("target", {})
        referenced.add(source.get("node_id"))
        referenced.add(target.get("node_id"))
    
    referenced.discard(None)
    
    # Check each equipment
    for eq in merged.get("nodes", {}).get("equipment", []):
        eq_id = eq.get("id")
        eq_tag = eq.get("tag")
        has_connections = eq.get("connections", {}).get("inlet_from") or eq.get("connections", {}).get("outlet_to")
        is_referenced = eq_id in referenced or eq_tag in referenced
        
        if not has_connections and not is_referenced:
            orphans.append(eq_id or eq_tag)
    
    return orphans


def load_ocr_data(ocr_path: str) -> list:
    """Load OCR extractions from a JSON file."""
    if not ocr_path or not Path(ocr_path).exists():
        return []
    
    with open(ocr_path) as f:
        data = json.load(f)
    
    # Handle both global and per-tile OCR formats
    if "all_extractions" in data:
        return data["all_extractions"]
    elif "extractions" in data:
        return data["extractions"]
    else:
        return []


def validate_against_ocr(merged: dict, ocr_data: list, fuzzy_threshold: float = 0.8) -> dict:
    """
    Validate extracted entities against OCR data to detect hallucinations.
    
    Returns dict with:
    - validated: entities found in OCR
    - unvalidated: entities NOT found in OCR (potential hallucinations)
    - ocr_only: OCR detections not matched to any entity
    """
    if not ocr_data:
        return {"validated": [], "unvalidated": [], "ocr_only": [], "warning": "No OCR data provided"}
    
    # Build OCR text index (uppercase for matching)
    ocr_texts = {}
    for ocr in ocr_data:
        text = ocr.get("text", "").strip().upper()
        if text and len(text) >= 2:
            ocr_texts[text] = ocr
    
    validated = []
    unvalidated = []
    matched_ocr_texts = set()
    
    # Check all entity types
    for entity_type in ["equipment", "instruments", "valves"]:
        entities = merged.get("nodes", {}).get(entity_type, [])
        
        for entity in entities:
            tag = entity.get("tag", "")
            if not tag:
                continue
            
            tag_upper = tag.strip().upper()
            
            # Exact match
            if tag_upper in ocr_texts:
                entity["ocr_validated"] = True
                entity["ocr_confidence"] = ocr_texts[tag_upper].get("confidence", 0)
                validated.append({"type": entity_type, "tag": tag, "match": "exact"})
                matched_ocr_texts.add(tag_upper)
                continue
            
            # Partial/fuzzy match using similarity ratio
            found = False
            for ocr_text in ocr_texts:
                # Calculate similarity ratio
                shorter = min(len(tag_upper), len(ocr_text))
                longer = max(len(tag_upper), len(ocr_text))
                if shorter == 0:
                    continue
                # Check substring match OR length-based similarity
                matches = sum(1 for a, b in zip(tag_upper, ocr_text) if a == b)
                similarity = matches / longer if longer > 0 else 0
                
                if (tag_upper in ocr_text or ocr_text in tag_upper or 
                    similarity >= fuzzy_threshold):
                    entity["ocr_validated"] = True
                    entity["ocr_confidence"] = ocr_texts[ocr_text].get("confidence", 0)
                    entity["ocr_match_type"] = "partial"
                    validated.append({"type": entity_type, "tag": tag, "match": "partial", "ocr_text": ocr_text})
                    matched_ocr_texts.add(ocr_text)
                    found = True
                    break
            
            if not found:
                entity["ocr_validated"] = False
                unvalidated.append({
                    "type": entity_type, 
                    "tag": tag, 
                    "id": entity.get("id"),
                    "warning": "Tag not found in OCR - possible hallucination"
                })
    
    # Find unmatched OCR detections (could be missed entities)
    ocr_only = []
    for text, ocr in ocr_texts.items():
        if text not in matched_ocr_texts:
            ocr_type = ocr.get("type", "label")
            if ocr_type in ["equipment_tag", "instrument_tag", "valve_tag"]:
                ocr_only.append({
                    "text": ocr.get("text"),
                    "type": ocr_type,
                    "bbox": ocr.get("bbox"),
                    "confidence": ocr.get("confidence"),
                    "warning": "OCR detection not matched to any extracted entity"
                })
    
    return {
        "validated": validated,
        "unvalidated": unvalidated,
        "ocr_only": ocr_only,
        "stats": {
            "total_entities": len(validated) + len(unvalidated),
            "validated_count": len(validated),
            "unvalidated_count": len(unvalidated),
            "validation_rate": len(validated) / max(1, len(validated) + len(unvalidated)),
            "ocr_unmatched": len(ocr_only)
        }
    }


def find_nearest_ocr(entity_bbox: list, ocr_data: list, max_distance: int = 100) -> dict:
    """
    Find the nearest OCR detection to an entity's bounding box.
    Uses center-to-center distance.
    """
    if not entity_bbox or not ocr_data:
        return None
    
    # Calculate entity center
    ey_center = (entity_bbox[0] + entity_bbox[2]) / 2
    ex_center = (entity_bbox[1] + entity_bbox[3]) / 2
    
    nearest = None
    min_distance = float('inf')
    
    for ocr in ocr_data:
        ocr_bbox = ocr.get("bbox")
        if not ocr_bbox:
            continue
        
        # Calculate OCR center
        oy_center = (ocr_bbox[0] + ocr_bbox[2]) / 2
        ox_center = (ocr_bbox[1] + ocr_bbox[3]) / 2
        
        # Euclidean distance
        distance = ((ey_center - oy_center) ** 2 + (ex_center - ox_center) ** 2) ** 0.5
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest = {
                "text": ocr.get("text"),
                "bbox": ocr_bbox,
                "confidence": ocr.get("confidence"),
                "distance": round(distance, 2)
            }
    
    return nearest


def calculate_bbox_overlap(bbox1: list, bbox2: list) -> float:
    """
    Calculate overlap percentage between two bboxes.
    Bboxes are [ymin, xmin, ymax, xmax] in normalized 0-1000 space.
    """
    y1_min, x1_min, y1_max, x1_max = bbox1
    y2_min, x2_min, y2_max, x2_max = bbox2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Return overlap as fraction of smaller bbox
    min_area = min(area1, area2)
    return intersection / min_area


def transform_coordinates(bbox: list, tile_norm_bbox: dict) -> list:
    """
    Transform tile-local normalized coordinates to global coordinates.
    
    Args:
        bbox: [ymin, xmin, ymax, xmax] in tile's 0-1000 space
        tile_norm_bbox: Tile's position in global 0-1000 space
    
    Returns:
        Transformed bbox in global 0-1000 space
    """
    tile_width = tile_norm_bbox["x2"] - tile_norm_bbox["x1"]
    tile_height = tile_norm_bbox["y2"] - tile_norm_bbox["y1"]
    
    ymin, xmin, ymax, xmax = bbox
    
    # Scale and translate
    global_xmin = tile_norm_bbox["x1"] + (xmin / 1000) * tile_width
    global_xmax = tile_norm_bbox["x1"] + (xmax / 1000) * tile_width
    global_ymin = tile_norm_bbox["y1"] + (ymin / 1000) * tile_height
    global_ymax = tile_norm_bbox["y1"] + (ymax / 1000) * tile_height
    
    return [int(global_ymin), int(global_xmin), int(global_ymax), int(global_xmax)]


def calculate_waypoint_similarity(waypoints1: list, waypoints2: list, tolerance: int = 50) -> float:
    """
    Calculate similarity between two waypoint paths.
    Uses average minimum distance between points.
    
    Returns:
        Similarity score 0.0-1.0 (1.0 = identical paths)
    """
    if not waypoints1 or not waypoints2:
        return 0.0
    
    def point_distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    # Calculate average minimum distance from wp1 to wp2
    total_dist = 0
    for p1 in waypoints1:
        min_dist = min(point_distance(p1, p2) for p2 in waypoints2)
        total_dist += min_dist
    
    avg_dist = total_dist / len(waypoints1)
    
    # Convert to similarity score (0-1)
    # Within tolerance = high similarity, beyond 5x tolerance = 0 similarity
    if avg_dist <= tolerance:
        return 1.0 - (avg_dist / tolerance) * 0.5  # 0.5-1.0 range
    elif avg_dist <= tolerance * 5:
        return 0.5 - (avg_dist - tolerance) / (tolerance * 4) * 0.5  # 0-0.5 range
    else:
        return 0.0


def check_waypoint_orthogonality(waypoints: list, tolerance_degrees: float = 5.0) -> list:
    """
    Check if waypoints form orthogonal (horizontal/vertical) paths.
    
    Returns:
        List of warnings for non-orthogonal segments.
    """
    import math
    
    warnings = []
    if len(waypoints) < 2:
        return warnings
    
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > 0 and dy > 0:
            # Segment is diagonal
            angle = math.degrees(math.atan2(dy, dx))
            # Check if close to 0, 45, or 90 degrees
            if angle > tolerance_degrees and angle < (90 - tolerance_degrees):
                warnings.append({
                    "segment": [i, i + 1],
                    "points": [waypoints[i], waypoints[i + 1]],
                    "angle_degrees": round(angle, 1),
                    "issue": "Non-orthogonal segment detected"
                })
    
    return warnings


def propagate_labels_bfs(lines: list, tolerance: int = 30) -> list:
    """
    Propagate line labels along connected segments using BFS.
    
    As described in Digitize-PID paper:
    1. Build adjacency graph from line endpoints
    2. For lines with labels, BFS to propagate to connected unlabeled lines
    3. Returns lines with propagated labels
    
    Args:
        lines: List of line dicts with route_waypoints
        tolerance: Distance threshold for considering endpoints connected
    
    Returns:
        Lines with label_propagated field where applicable
    """
    from collections import deque
    
    if not lines:
        return lines
    
    # Extract endpoints for each line
    endpoints = []
    for i, line in enumerate(lines):
        waypoints = line.get("route_waypoints", [])
        if len(waypoints) >= 2:
            endpoints.append({
                "line_idx": i,
                "start": tuple(waypoints[0]),
                "end": tuple(waypoints[-1])
            })
    
    # Build adjacency: line indices that share endpoints
    def points_close(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5 <= tolerance
    
    adjacency = {i: set() for i in range(len(lines))}
    
    for i, ep1 in enumerate(endpoints):
        for j, ep2 in enumerate(endpoints):
            if i >= j:
                continue
            # Check if any endpoints are close
            if (points_close(ep1["start"], ep2["start"]) or
                points_close(ep1["start"], ep2["end"]) or
                points_close(ep1["end"], ep2["start"]) or
                points_close(ep1["end"], ep2["end"])):
                adjacency[ep1["line_idx"]].add(ep2["line_idx"])
                adjacency[ep2["line_idx"]].add(ep1["line_idx"])
    
    # Find labeled lines (have line_number in specifications or tag)
    labeled_indices = set()
    for i, line in enumerate(lines):
        spec = line.get("specifications", {})
        if spec.get("line_number") or line.get("tag"):
            labeled_indices.add(i)
    
    # BFS from each labeled line
    propagated_from = {}  # line_idx -> source line_idx
    
    for start_idx in labeled_indices:
        source_label = (lines[start_idx].get("specifications", {}).get("line_number") or 
                       lines[start_idx].get("tag"))
        
        visited = {start_idx}
        queue = deque([start_idx])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Check if neighbor is unlabeled
                    neighbor_spec = lines[neighbor].get("specifications", {})
                    if not neighbor_spec.get("line_number") and not lines[neighbor].get("tag"):
                        # Propagate label
                        if neighbor not in propagated_from:
                            propagated_from[neighbor] = start_idx
                            queue.append(neighbor)
    
    # Apply propagated labels
    for line_idx, source_idx in propagated_from.items():
        source_label = (lines[source_idx].get("specifications", {}).get("line_number") or 
                       lines[source_idx].get("tag"))
        if source_label:
            if "specifications" not in lines[line_idx]:
                lines[line_idx]["specifications"] = {}
            lines[line_idx]["specifications"]["line_number_propagated"] = source_label
            lines[line_idx]["label_propagated"] = True
            lines[line_idx]["propagated_from"] = lines[source_idx].get("id")
    
    return lines


def deduplicate_lines_by_geometry(lines: list, similarity_threshold: float = 0.7) -> list:
    """
    Deduplicate lines using geometry-first approach.
    
    Priority:
    1. Waypoint path similarity > threshold -> keep higher confidence
    2. Same source/target nodes -> keep higher confidence
    3. Tag/ID match -> keep higher confidence (secondary)
    """
    if not lines:
        return []
    
    # Sort by confidence (highest first)
    sorted_lines = sorted(lines, key=lambda x: x.get("confidence", 0), reverse=True)
    
    kept = []
    for line in sorted_lines:
        is_duplicate = False
        line_waypoints = line.get("route_waypoints", [])
        line_source = line.get("source", {}).get("node_id")
        line_target = line.get("target", {}).get("node_id")
        line_id = line.get("id") or line.get("tag")
        
        for existing in kept:
            existing_waypoints = existing.get("route_waypoints", [])
            existing_source = existing.get("source", {}).get("node_id")
            existing_target = existing.get("target", {}).get("node_id")
            existing_id = existing.get("id") or existing.get("tag")
            
            # Check 1: Waypoint path similarity (geometry-first)
            if line_waypoints and existing_waypoints:
                similarity = calculate_waypoint_similarity(line_waypoints, existing_waypoints)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            # Check 2: Same source and target nodes
            if (line_source and line_target and 
                line_source == existing_source and 
                line_target == existing_target):
                is_duplicate = True
                break
            
            # Check 3: Exact ID/tag match (secondary)
            if line_id and line_id == existing_id:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(line)
    
    return kept


def deduplicate_entities(entities: list, id_field: str = "id", overlap_threshold: float = 0.95, audit_log: list = None) -> list:
    """
    Remove duplicate entities based on tag matching and spatial overlap.
    
    Priority:
    1. Exact tag match -> keep entity with higher confidence
    2. Spatial overlap > threshold -> keep entity with higher confidence
    
    IMPORTANT: Entities with DIFFERENT non-null tags are NEVER merged, even with high overlap.
    This prevents losing distinct valves that happen to be near each other.
    
    Args:
        audit_log: Optional list to append dropped entity info for debugging
    """
    if not entities:
        return []
    
    keyed_entities = []
    unkeyed_entities = []

    for entity in entities:
        tag = entity.get("tag")
        entity_id = entity.get(id_field)
        if tag or entity_id:
            keyed_entities.append(entity)
        else:
            unkeyed_entities.append(entity)

    by_tag = defaultdict(list)
    for entity in keyed_entities:
        tag = entity.get("tag") or entity.get(id_field)
        by_tag[tag].append(entity)
    
    deduplicated = []
    
    for tag, group in by_tag.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Multiple entities with same tag - keep highest confidence
            sorted_group = sorted(group, key=lambda x: x.get("confidence", 0), reverse=True)
            best = sorted_group[0]
            
            # Check if others have significant spatial separation (could be different items)
            for other in sorted_group[1:]:
                if "bbox" in best and "bbox" in other:
                    overlap = calculate_bbox_overlap(best["bbox"], other["bbox"])
                    if overlap < overlap_threshold:
                        # Spatially distinct - keep both but rename
                        other[id_field] = f"{other.get(id_field, 'unknown')}_dup"
                        deduplicated.append(other)
                    elif audit_log is not None:
                        audit_log.append({
                            "action": "merged_duplicate",
                            "kept": best.get("tag") or best.get(id_field),
                            "dropped": other.get("tag") or other.get(id_field),
                            "overlap": round(overlap, 3),
                            "reason": f"same tag '{tag}' with {overlap:.0%} overlap"
                        })
            
            deduplicated.append(best)

    if unkeyed_entities:
        sorted_unkeyed = sorted(unkeyed_entities, key=lambda x: x.get("confidence", 0), reverse=True)
        kept_unkeyed = []

        for entity in sorted_unkeyed:
            is_duplicate = False
            entity_bbox = entity.get("bbox")

            for existing in kept_unkeyed:
                existing_bbox = existing.get("bbox")
                if entity_bbox and existing_bbox:
                    overlap = calculate_bbox_overlap(entity_bbox, existing_bbox)
                    if overlap >= overlap_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept_unkeyed.append(entity)

        deduplicated.extend(kept_unkeyed)

    candidates = deduplicated
    candidates_sorted = sorted(
        candidates,
        key=lambda e: (
            1 if e.get("tag") else 0,
            1 if e.get(id_field) else 0,
            e.get("confidence", 0)
        ),
        reverse=True
    )

    spatially_deduped = []
    for entity in candidates_sorted:
        entity_bbox = entity.get("bbox")
        entity_tag = entity.get("tag")
        if not entity_bbox:
            spatially_deduped.append(entity)
            continue

        is_duplicate = False
        for existing in spatially_deduped:
            existing_bbox = existing.get("bbox")
            existing_tag = existing.get("tag")
            
            # CRITICAL: Never merge entities with DIFFERENT non-null tags
            # Two valves V-100 and V-101 near each other are NOT duplicates
            if entity_tag and existing_tag and entity_tag != existing_tag:
                continue  # Skip this comparison, they're different entities
            
            if existing_bbox and calculate_bbox_overlap(entity_bbox, existing_bbox) >= overlap_threshold:
                is_duplicate = True
                if audit_log is not None:
                    audit_log.append({
                        "action": "spatial_dedup",
                        "kept": existing_tag or existing.get(id_field),
                        "dropped": entity_tag or entity.get(id_field),
                        "overlap": round(calculate_bbox_overlap(entity_bbox, existing_bbox), 3),
                        "reason": "high spatial overlap with no distinguishing tag"
                    })
                break

        if not is_duplicate:
            spatially_deduped.append(entity)

    return spatially_deduped


def merge_json_files(json_paths: list, tiles_metadata_path: str = None, 
                     min_confidence: float = 0.0) -> dict:
    """
    Merge multiple tile extraction JSONs.
    
    Args:
        json_paths: List of paths to tile JSON files
        tiles_metadata_path: Path to tiles_metadata.json for coordinate transformation
        min_confidence: Minimum confidence threshold (0.0-1.0) to include entities
    
    Returns:
        Merged and deduplicated JSON
    """
    # Load tiles metadata if available
    tile_info = {}
    if tiles_metadata_path:
        with open(tiles_metadata_path) as f:
            metadata = json.load(f)
            for tile in metadata.get("tiles", []):
                tile_info[tile["filename"].replace(".png", ".json")] = tile["normalized_bbox"]
    
    # Initialize merged structure
    merged = {
        "schema_version": "1.1.1",
        "extraction_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_image": None,
        "metadata": {},
        "legend": {"symbols": [], "line_types": [], "abbreviations": []},
        "nodes": {"equipment": [], "instruments": [], "valves": []},
        "edges": {"process_lines": [], "signal_lines": []},
        "process_logic": {"flow_sequences": [], "control_relationships": [], "interlocks": []},
        "control_loops": [],
        "utilities": [],
        "validation": {"warnings": []}
    }

    if tiles_metadata_path and not merged["source_image"]:
        try:
            merged["source_image"] = metadata.get("source_image")
        except Exception:
            pass
    
    # Collect all entities
    all_equipment = []
    all_instruments = []
    all_valves = []
    all_lines = []
    all_signal_lines = []
    all_utilities = []
    
    for json_path in json_paths:
        path = Path(json_path)
        if not path.exists():
            print(f"Warning: File not found: {json_path}")
            continue
        
        with open(path) as f:
            data = json.load(f)
        
        # Get tile transformation info
        tile_bbox = tile_info.get(path.name, {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000})
        
        # Collect metadata (prefer first non-null)
        if not merged["metadata"] and data.get("metadata"):
            merged["metadata"] = data["metadata"]
        
        if not merged["source_image"] and data.get("source_image"):
            merged["source_image"] = data["source_image"]
        
        # Collect and transform entities (with confidence filtering)
        nodes = data.get("nodes", {})
        
        for eq in nodes.get("equipment", []):
            if eq.get("confidence", 1.0) < min_confidence:
                continue
            if "bbox" in eq:
                eq["bbox"] = transform_coordinates(eq["bbox"], tile_bbox)
            all_equipment.append(eq)
        
        for inst in nodes.get("instruments", []):
            if inst.get("confidence", 1.0) < min_confidence:
                continue
            if "bbox" in inst:
                inst["bbox"] = transform_coordinates(inst["bbox"], tile_bbox)
            all_instruments.append(inst)
        
        for valve in nodes.get("valves", []):
            if valve.get("confidence", 1.0) < min_confidence:
                continue
            if "bbox" in valve:
                valve["bbox"] = transform_coordinates(valve["bbox"], tile_bbox)
            all_valves.append(valve)
        
        edges = data.get("edges", {})
        for line in edges.get("process_lines", []):
            # Transform waypoints
            if "route_waypoints" in line:
                line["route_waypoints"] = [
                    [
                        int(tile_bbox["x1"] + (x / 1000) * (tile_bbox["x2"] - tile_bbox["x1"])),
                        int(tile_bbox["y1"] + (y / 1000) * (tile_bbox["y2"] - tile_bbox["y1"]))
                    ]
                    for x, y in line["route_waypoints"]
                ]
            all_lines.append(line)
        
        # Transform signal_lines waypoints (same logic as process_lines)
        for sig_line in edges.get("signal_lines", []):
            if "route_waypoints" in sig_line:
                sig_line["route_waypoints"] = [
                    [
                        int(tile_bbox["x1"] + (x / 1000) * (tile_bbox["x2"] - tile_bbox["x1"])),
                        int(tile_bbox["y1"] + (y / 1000) * (tile_bbox["y2"] - tile_bbox["y1"]))
                    ]
                    for x, y in sig_line["route_waypoints"]
                ]
            all_signal_lines.append(sig_line)
        
        for util in data.get("utilities", []):
            if "position" in util:
                util["position"]["x"] = int(tile_bbox["x1"] + (util["position"]["x"] / 1000) * (tile_bbox["x2"] - tile_bbox["x1"]))
                util["position"]["y"] = int(tile_bbox["y1"] + (util["position"]["y"] / 1000) * (tile_bbox["y2"] - tile_bbox["y1"]))
            all_utilities.append(util)
        
        # Merge legends
        legend = data.get("legend", {})
        merged["legend"]["symbols"].extend(legend.get("symbols", []))
        merged["legend"]["line_types"].extend(legend.get("line_types", []))
        merged["legend"]["abbreviations"].extend(legend.get("abbreviations", []))
        
        # Merge process logic
        logic = data.get("process_logic", {})
        merged["process_logic"]["flow_sequences"].extend(logic.get("flow_sequences", []))
        merged["process_logic"]["control_relationships"].extend(logic.get("control_relationships", []))
        merged["process_logic"]["interlocks"].extend(logic.get("interlocks", []))

        merged["control_loops"].extend(data.get("control_loops", []))
    
    # Deduplicate with optional audit logging
    audit_log = []
    merged["nodes"]["equipment"] = deduplicate_entities(all_equipment, "id", audit_log=audit_log)
    merged["nodes"]["instruments"] = deduplicate_entities(all_instruments, "id", audit_log=audit_log)
    merged["nodes"]["valves"] = deduplicate_entities(all_valves, "id", audit_log=audit_log)
    merged["edges"]["process_lines"] = deduplicate_lines_by_geometry(all_lines)
    merged["edges"]["signal_lines"] = deduplicate_lines_by_geometry(all_signal_lines)
    merged["utilities"] = deduplicate_entities(all_utilities, "id", audit_log=audit_log)
    
    # Store audit log in validation section
    if audit_log:
        merged["validation"]["dedup_audit"] = audit_log
    
    # BFS label propagation (Digitize-PID paper technique)
    merged["edges"]["process_lines"] = propagate_labels_bfs(merged["edges"]["process_lines"])
    merged["edges"]["signal_lines"] = propagate_labels_bfs(merged["edges"]["signal_lines"])
    
    # Assign unique IDs to entities that don't have them
    reset_id_counters()
    for eq in merged["nodes"]["equipment"]:
        if not eq.get("id"):
            eq["id"] = generate_unique_id("equipment")
    for inst in merged["nodes"]["instruments"]:
        if not inst.get("id"):
            inst["id"] = generate_unique_id("instrument")
    for valve in merged["nodes"]["valves"]:
        if not valve.get("id"):
            valve["id"] = generate_unique_id("valve")
    for line in merged["edges"]["process_lines"]:
        if not line.get("id"):
            line["id"] = generate_unique_id("line")

    for line in merged["edges"]["signal_lines"]:
        if not line.get("id"):
            line["id"] = generate_unique_id("signal_line")
    
    # Deduplicate legends
    seen_symbols = set()
    unique_symbols = []
    for sym in merged["legend"]["symbols"]:
        key = sym.get("symbol_id", str(sym))
        if key not in seen_symbols:
            seen_symbols.add(key)
            unique_symbols.append(sym)
    merged["legend"]["symbols"] = unique_symbols
    
    # Run validation checks
    validation_result = validate_references(merged)
    orphans = detect_orphans(merged)
    
    # Run orthogonality checks on all lines
    orthogonality_warnings = []
    for line in merged["edges"]["process_lines"]:
        if line.get("route_waypoints"):
            orth_issues = check_waypoint_orthogonality(line["route_waypoints"])
            for issue in orth_issues:
                issue["line_id"] = line.get("id")
                orthogonality_warnings.append(issue)
    
    for sig_line in merged["edges"]["signal_lines"]:
        if sig_line.get("route_waypoints"):
            orth_issues = check_waypoint_orthogonality(sig_line["route_waypoints"])
            for issue in orth_issues:
                issue["line_id"] = sig_line.get("id")
                orthogonality_warnings.append(issue)
    
    # Update validation section with structured output
    merged["validation"]["total_equipment"] = len(merged["nodes"]["equipment"])
    merged["validation"]["total_instruments"] = len(merged["nodes"]["instruments"])
    merged["validation"]["total_valves"] = len(merged["nodes"]["valves"])
    merged["validation"]["total_lines"] = len(merged["edges"]["process_lines"]) + len(merged["edges"]["signal_lines"])
    merged["validation"]["orphaned_nodes"] = orphans
    merged["validation"]["unresolved_references"] = validation_result["unresolved_references"]
    merged["validation"]["incomplete_loops"] = validation_result["incomplete_loops"]
    merged["validation"]["orthogonality_warnings"] = orthogonality_warnings
    merged["validation"]["warnings"] = validation_result["warnings"]
    merged["validation"]["is_valid"] = (
        len(validation_result["unresolved_references"]) == 0 and 
        len(orphans) == 0 and
        len(validation_result["incomplete_loops"]) == 0
    )
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge P&ID tile extraction JSONs into consolidated output"
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Paths to tile JSON files to merge"
    )
    parser.add_argument(
        "--tiles-metadata", "-t",
        help="Path to tiles_metadata.json for coordinate transformation"
    )
    parser.add_argument(
        "--output", "-o",
        default="merged_output.json",
        help="Output file path (default: merged_output.json)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Print detailed validation warnings"
    )
    parser.add_argument(
        "--min-confidence", "-c",
        type=float, default=0.0,
        help="Minimum confidence threshold to include entities (0.0-1.0)"
    )
    parser.add_argument(
        "--ocr-file",
        help="Path to OCR global JSON for hallucination detection (ocr_global.json)"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print deduplication audit log showing what was merged/dropped"
    )
    
    args = parser.parse_args()
    
    merged = merge_json_files(args.json_files, args.tiles_metadata, args.min_confidence)
    
    # OCR validation if provided
    ocr_validation = None
    if args.ocr_file:
        ocr_data = load_ocr_data(args.ocr_file)
        if ocr_data:
            ocr_validation = validate_against_ocr(merged, ocr_data)
            merged["validation"]["ocr_validation"] = ocr_validation["stats"]
            merged["validation"]["potential_hallucinations"] = ocr_validation["unvalidated"]
            merged["validation"]["missed_ocr_tags"] = ocr_validation["ocr_only"]
    
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    
    print(f"Merged {len(args.json_files)} files -> {args.output}")
    print(f"  Equipment: {merged['validation']['total_equipment']}")
    print(f"  Instruments: {merged['validation']['total_instruments']}")
    print(f"  Valves: {merged['validation']['total_valves']}")
    print(f"  Lines: {merged['validation']['total_lines']}")
    
    # Validation summary
    is_valid = merged['validation']['is_valid']
    orphans = merged['validation'].get('orphaned_nodes', [])
    warnings = merged['validation'].get('warnings', [])
    
    if is_valid:
        print("  Validation: [PASSED]")
    else:
        print(f"  Validation: [FAILED] ({len(warnings)} warnings, {len(orphans)} orphans)")
        
        if args.validate:
            if warnings:
                print("\n  === Reference Warnings ===")
                for w in warnings[:10]:  # Limit output
                    print(f"    - {w}")
                if len(warnings) > 10:
                    print(f"    ... and {len(warnings) - 10} more")
            
            if orphans:
                print("\n  === Orphaned Nodes ===")
                for o in orphans[:10]:
                    print(f"    - {o}")
                if len(orphans) > 10:
                    print(f"    ... and {len(orphans) - 10} more")
    
    # OCR validation summary
    if ocr_validation:
        stats = ocr_validation["stats"]
        rate = stats["validation_rate"] * 100
        print(f"\n  === OCR Validation ===")
        print(f"  OCR Validated: {stats['validated_count']}/{stats['total_entities']} ({rate:.0f}%)")
        
        if stats["unvalidated_count"] > 0:
            print(f"  ⚠️  Potential Hallucinations: {stats['unvalidated_count']}")
            if args.validate:
                for item in ocr_validation["unvalidated"][:5]:
                    print(f"      - {item['type']}: {item['tag']}")
                if stats["unvalidated_count"] > 5:
                    print(f"      ... and {stats['unvalidated_count'] - 5} more")
        
        if stats["ocr_unmatched"] > 0:
            print(f"  ℹ️  OCR tags not extracted: {stats['ocr_unmatched']}")
            if args.validate:
                for item in ocr_validation["ocr_only"][:5]:
                    print(f"      - {item['type']}: {item['text']}")


if __name__ == "__main__":
    main()
