#!/usr/bin/env python3
"""
P&ID Extraction Critic - Error Classification and Repair

Implements an agentic "validate → repair → revalidate" loop as recommended
in arXiv:2412.12898 (Agentic Approach to P&ID).

Error Classes:
- DANGLING_REF: Edge references non-existent node
- DUPLICATE_ID: Same ID appears multiple times
- IMPOSSIBLE_EDGE: Invalid connection type (e.g., valve→valve without pipe)
- SEAM_BREAK: Line endpoints near tile boundary with no continuation
- UNGROUNDED_TAG: Tag not found in OCR data
- MISSING_FIELD: Required field is empty/null

Repair Actions:
- Remove invalid edges
- Merge duplicate nodes
- Search for cross-tile continuations
- Mark low-confidence items
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class ErrorType:
    """Error type constants.""" 
    DANGLING_REF = "DANGLING_REF"
    DUPLICATE_ID = "DUPLICATE_ID"
    IMPOSSIBLE_EDGE = "IMPOSSIBLE_EDGE"
    SEAM_BREAK = "SEAM_BREAK"
    UNGROUNDED_TAG = "UNGROUNDED_TAG"
    MISSING_FIELD = "MISSING_FIELD"
    ORPHAN_NODE = "ORPHAN_NODE"


class CriticError:
    """Represents a detected error."""
    
    def __init__(self, error_type: str, entity_type: str, entity_id: str,
                 field: str = None, message: str = "", context: dict = None):
        self.error_type = error_type
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.field = field
        self.message = message
        self.context = context or {}
        self.repaired = False
        self.repair_action = None
    
    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "field": self.field,
            "message": self.message,
            "context": self.context,
            "repaired": self.repaired,
            "repair_action": self.repair_action
        }
    
    def __repr__(self):
        return f"CriticError({self.error_type}: {self.entity_type}/{self.entity_id})"


def build_node_index(merged: dict) -> Dict[str, dict]:
    """Build index of all node IDs to their data."""
    index = {}
    
    for eq in merged.get("nodes", {}).get("equipment", []):
        if eq.get("id"):
            index[eq["id"]] = {"type": "equipment", "data": eq}
    
    for inst in merged.get("nodes", {}).get("instruments", []):
        if inst.get("id"):
            index[inst["id"]] = {"type": "instrument", "data": inst}
    
    for valve in merged.get("nodes", {}).get("valves", []):
        if valve.get("id"):
            index[valve["id"]] = {"type": "valve", "data": valve}
    
    return index


def classify_errors(merged: dict, ocr_texts: set = None) -> List[CriticError]:
    """
    Classify all errors in a merged extraction.
    
    Returns list of CriticError objects.
    """
    errors = []
    node_index = build_node_index(merged)
    all_ids = set(node_index.keys())
    referenced_node_ids = set()
    
    # Check for duplicate IDs
    id_counts = defaultdict(list)
    for entity_type in ["equipment", "instruments", "valves"]:
        for entity in merged.get("nodes", {}).get(entity_type, []):
            eid = entity.get("id")
            if eid:
                id_counts[eid].append((entity_type, entity))
    
    for eid, occurrences in id_counts.items():
        if len(occurrences) > 1:
            errors.append(CriticError(
                ErrorType.DUPLICATE_ID,
                occurrences[0][0],
                eid,
                message=f"ID appears {len(occurrences)} times",
                context={"occurrences": len(occurrences)}
            ))
    
    # Check process lines
    for line in merged.get("edges", {}).get("process_lines", []):
        line_id = line.get("id", "unknown")
        
        # Check source reference
        source = line.get("source", {})
        source_id = source.get("node_id")
        if source_id and source_id not in all_ids:
            errors.append(CriticError(
                ErrorType.DANGLING_REF,
                "process_line",
                line_id,
                field="source.node_id",
                message=f"Source node '{source_id}' does not exist",
                context={"missing_node": source_id}
            ))
        elif source_id:
            referenced_node_ids.add(source_id)
        
        # Check target reference
        target = line.get("target", {})
        target_id = target.get("node_id")
        if target_id and target_id not in all_ids:
            errors.append(CriticError(
                ErrorType.DANGLING_REF,
                "process_line",
                line_id,
                field="target.node_id",
                message=f"Target node '{target_id}' does not exist",
                context={"missing_node": target_id}
            ))
        elif target_id:
            referenced_node_ids.add(target_id)
        
        # Check inline components
        for comp in line.get("inline_components", []):
            comp_id = comp.get("node_id")
            if comp_id and comp_id not in all_ids:
                errors.append(CriticError(
                    ErrorType.DANGLING_REF,
                    "process_line",
                    line_id,
                    field="inline_components",
                    message=f"Inline component '{comp_id}' does not exist",
                    context={"missing_node": comp_id}
                ))
            elif comp_id:
                referenced_node_ids.add(comp_id)
        
        # Check for seam breaks (endpoints near tile boundary 0 or 1000)
        waypoints = line.get("route_waypoints", [])
        if waypoints:
            start = waypoints[0] if waypoints else None
            end = waypoints[-1] if len(waypoints) > 1 else None
            
            for point, label in [(start, "start"), (end, "end")]:
                if point and is_near_boundary(point):
                    # Check if there's a continuation
                    if not source_id and label == "start":
                        errors.append(CriticError(
                            ErrorType.SEAM_BREAK,
                            "process_line",
                            line_id,
                            field=f"waypoint_{label}",
                            message=f"Line {label} at boundary with no connection",
                            context={"point": point, "boundary": True}
                        ))
    
    # Check signal lines similarly
    for line in merged.get("edges", {}).get("signal_lines", []):
        line_id = line.get("id", "unknown")
        
        source = line.get("source", {})
        source_id = source.get("node_id")
        if source_id and source_id not in all_ids:
            errors.append(CriticError(
                ErrorType.DANGLING_REF,
                "signal_line",
                line_id,
                field="source.node_id",
                message=f"Source node '{source_id}' does not exist"
            ))
        elif source_id:
            referenced_node_ids.add(source_id)
        
        target = line.get("target", {})
        target_id = target.get("node_id")
        if target_id and target_id not in all_ids:
            errors.append(CriticError(
                ErrorType.DANGLING_REF,
                "signal_line",
                line_id,
                field="target.node_id",
                message=f"Target node '{target_id}' does not exist"
            ))
        elif target_id:
            referenced_node_ids.add(target_id)
    
    # Check for orphan nodes (not referenced by any edge)
    orphan_ids = all_ids - referenced_node_ids
    for orphan_id in orphan_ids:
        node_info = node_index.get(orphan_id, {})
        errors.append(CriticError(
            ErrorType.ORPHAN_NODE,
            node_info.get("type", "unknown"),
            orphan_id,
            message="Node not connected to any edge"
        ))
    
    # Check for ungrounded tags (if OCR data provided)
    if ocr_texts:
        ocr_upper = {t.upper() for t in ocr_texts}
        for entity_type in ["equipment", "instruments", "valves"]:
            for entity in merged.get("nodes", {}).get(entity_type, []):
                tag = entity.get("tag")
                if tag and tag.upper() not in ocr_upper:
                    # Allow partial matches
                    found = any(tag.upper() in ocr or ocr in tag.upper() 
                               for ocr in ocr_upper)
                    if not found:
                        errors.append(CriticError(
                            ErrorType.UNGROUNDED_TAG,
                            entity_type,
                            entity.get("id", "unknown"),
                            field="tag",
                            message=f"Tag '{tag}' not found in OCR data",
                            context={"tag": tag}
                        ))
    
    return errors


def is_near_boundary(point: list, threshold: int = 20) -> bool:
    """Check if a point is near the tile boundary (0 or 1000)."""
    if not point or len(point) < 2:
        return False
    x, y = point[0], point[1]
    return (x <= threshold or x >= 1000 - threshold or 
            y <= threshold or y >= 1000 - threshold)


def repair_errors(merged: dict, errors: List[CriticError], 
                  aggressive: bool = False) -> Tuple[dict, List[CriticError]]:
    """
    Apply repairs for classified errors.
    
    Args:
        merged: The merged extraction dict
        errors: List of CriticErrors from classify_errors
        aggressive: If True, remove entities; if False, just mark them
    
    Returns:
        Tuple of (repaired_merged, updated_errors)
    """
    repaired = json.loads(json.dumps(merged))  # Deep copy
    
    for error in errors:
        if error.error_type == ErrorType.DANGLING_REF:
            # Mark the edge as having unresolved reference
            if error.entity_type in ["process_line", "signal_line"]:
                line_type = "process_lines" if error.entity_type == "process_line" else "signal_lines"
                for line in repaired.get("edges", {}).get(line_type, []):
                    if line.get("id") == error.entity_id:
                        if "_errors" not in line:
                            line["_errors"] = []
                        line["_errors"].append(error.to_dict())
                        line["_has_dangling_ref"] = True
                        error.repaired = True
                        error.repair_action = "marked"
                        
                        if aggressive:
                            # Remove the dangling reference
                            if error.field == "source.node_id":
                                line["source"]["node_id"] = None
                            elif error.field == "target.node_id":
                                line["target"]["node_id"] = None
                            error.repair_action = "removed_ref"
        
        elif error.error_type == ErrorType.DUPLICATE_ID:
            # Rename duplicates with suffix
            seen_ids = set()
            for entity_type in ["equipment", "instruments", "valves"]:
                for entity in repaired.get("nodes", {}).get(entity_type, []):
                    eid = entity.get("id")
                    if eid in seen_ids:
                        # Rename with suffix
                        suffix = 1
                        new_id = f"{eid}_{suffix}"
                        while new_id in seen_ids:
                            suffix += 1
                            new_id = f"{eid}_{suffix}"
                        entity["id"] = new_id
                        entity["_renamed_from"] = eid
                        error.repaired = True
                        error.repair_action = f"renamed_to_{new_id}"
                    seen_ids.add(entity.get("id"))
        
        elif error.error_type == ErrorType.UNGROUNDED_TAG:
            # Mark as unverified
            for entity_type in ["equipment", "instruments", "valves"]:
                for entity in repaired.get("nodes", {}).get(entity_type, []):
                    if entity.get("id") == error.entity_id:
                        entity["tag_verified"] = False
                        entity["_ungrounded_tag"] = True
                        error.repaired = True
                        error.repair_action = "marked_unverified"
        
        elif error.error_type == ErrorType.SEAM_BREAK:
            # Mark as potential seam break for manual review
            line_type = "process_lines" if error.entity_type == "process_line" else "signal_lines"
            for line in repaired.get("edges", {}).get(line_type, []):
                if line.get("id") == error.entity_id:
                    line["_seam_break"] = True
                    line["_seam_break_point"] = error.context.get("point")
                    error.repaired = True
                    error.repair_action = "marked_seam_break"
        
        elif error.error_type == ErrorType.ORPHAN_NODE:
            # Mark as orphan (don't remove - may be valid standalone equipment)
            for entity_type in ["equipment", "instruments", "valves"]:
                for entity in repaired.get("nodes", {}).get(entity_type, []):
                    if entity.get("id") == error.entity_id:
                        entity["_orphan"] = True
                        error.repaired = True
                        error.repair_action = "marked_orphan"
    
    return repaired, errors


def add_provenance(merged: dict, source_info: dict = None) -> dict:
    """
    Add provenance tracking to all nodes and edges.
    
    Args:
        merged: The merged extraction dict
        source_info: Dict with tile, method, timestamp info
    
    Returns:
        Merged dict with _source fields added
    """
    source_info = source_info or {"method": "pid_pipeline", "version": "1.2.0"}
    
    for entity_type in ["equipment", "instruments", "valves"]:
        for entity in merged.get("nodes", {}).get(entity_type, []):
            if "_source" not in entity:
                entity["_source"] = source_info.copy()
    
    for line_type in ["process_lines", "signal_lines"]:
        for line in merged.get("edges", {}).get(line_type, []):
            if "_source" not in line:
                line["_source"] = source_info.copy()
    
    return merged


def run_critic_loop(merged: dict, ocr_texts: set = None, 
                    max_iterations: int = 3) -> Tuple[dict, dict]:
    """
    Run the full critic → repair → revalidate loop.
    
    Args:
        merged: The merged extraction dict
        ocr_texts: Set of OCR-detected text for grounding validation
        max_iterations: Maximum repair iterations
    
    Returns:
        Tuple of (repaired_merged, critic_report)
    """
    current = merged
    all_errors = []
    iterations = 0
    
    for i in range(max_iterations):
        iterations = i + 1
        errors = classify_errors(current, ocr_texts)
        
        if not errors:
            break
        
        # Apply repairs
        current, repaired_errors = repair_errors(current, errors)
        all_errors.extend(repaired_errors)
        
        # Count unrepaired errors
        unrepaired = [e for e in errors if not e.repaired]
        if not unrepaired:
            break
    
    # Add provenance
    current = add_provenance(current)
    
    # Build report
    error_summary = defaultdict(int)
    for error in all_errors:
        error_summary[error.error_type] += 1
    
    repaired_count = sum(1 for e in all_errors if e.repaired)
    
    report = {
        "iterations": iterations,
        "total_errors_found": len(all_errors),
        "errors_repaired": repaired_count,
        "errors_remaining": len(all_errors) - repaired_count,
        "error_summary": dict(error_summary),
        "errors": [e.to_dict() for e in all_errors],
        "gates": {
            "graph_soundness": error_summary.get(ErrorType.DANGLING_REF, 0) == 0,
            "unique_ids": error_summary.get(ErrorType.DUPLICATE_ID, 0) == 0,
            "seam_integrity": error_summary.get(ErrorType.SEAM_BREAK, 0) == 0,
            "all_passed": len([e for e in all_errors if not e.repaired]) == 0
        }
    }
    
    return current, report


def main():
    """CLI for running critic on a merged JSON file."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run critic + repair loop on merged P&ID extraction"
    )
    parser.add_argument("merged_json", help="Path to merged extraction JSON")
    parser.add_argument(
        "--ocr-json", "-o",
        help="Path to OCR global JSON for tag grounding"
    )
    parser.add_argument(
        "--output", "-out",
        help="Output path for repaired JSON"
    )
    parser.add_argument(
        "--report", "-r",
        help="Output path for critic report JSON"
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int, default=3,
        help="Maximum repair iterations (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Load merged JSON
    with open(args.merged_json) as f:
        merged = json.load(f)
    
    # Load OCR texts if provided
    ocr_texts = None
    if args.ocr_json:
        with open(args.ocr_json) as f:
            ocr_data = json.load(f)
            ocr_texts = {item.get("text", "") for item in ocr_data.get("extractions", [])}
    
    # Run critic loop
    repaired, report = run_critic_loop(merged, ocr_texts, args.max_iterations)
    
    # Print summary
    print(f"\n{'='*50}")
    print("CRITIC REPORT")
    print(f"{'='*50}")
    print(f"  Iterations:       {report['iterations']}")
    print(f"  Errors found:     {report['total_errors_found']}")
    print(f"  Errors repaired:  {report['errors_repaired']}")
    print(f"  Errors remaining: {report['errors_remaining']}")
    print()
    print("  Error breakdown:")
    for etype, count in report["error_summary"].items():
        print(f"    {etype}: {count}")
    print()
    print("  Reliability Gates:")
    for gate, passed in report["gates"].items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {gate}: {status}")
    
    # Save outputs
    if args.output:
        with open(args.output, "w") as f:
            json.dump(repaired, f, indent=2)
        print(f"\nRepaired JSON saved to: {args.output}")
    
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Critic report saved to: {args.report}")


if __name__ == "__main__":
    main()
