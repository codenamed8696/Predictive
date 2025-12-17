#!/usr/bin/env python3
"""
P&ID Pipeline Evaluation Framework

Compares different pipeline configurations (A/B/C testing):
- A: LLM + OCR (baseline)
- B: LLM + OCR + CV lines/shapes
- C: B + BFS label propagation

Metrics tracked:
- OCR validation rate
- Unresolved references count
- Orphan nodes count
- Orthogonality warnings
- Label coverage (lines with labels)
- Total entities extracted

Usage:
    python evaluate_pipeline.py <merged_json_a> <merged_json_b> [<merged_json_c>]
    
Or for single file evaluation:
    python evaluate_pipeline.py <merged_json> --single
"""

import argparse
import json
import sys
from pathlib import Path


def evaluate_extraction(merged: dict) -> dict:
    """
    Evaluate a merged extraction JSON and return metrics.
    
    Includes edge-centric metrics as recommended in arXiv:2411.13929.
    """
    validation = merged.get("validation", {})
    nodes = merged.get("nodes", {})
    edges = merged.get("edges", {})
    
    # Count entities
    equipment_count = len(nodes.get("equipment", []))
    instrument_count = len(nodes.get("instruments", []))
    valve_count = len(nodes.get("valves", []))
    process_lines = edges.get("process_lines", [])
    signal_lines = edges.get("signal_lines", [])
    
    # Build node ID set for reference checking
    all_node_ids = set()
    for entity_type in ["equipment", "instruments", "valves"]:
        for entity in nodes.get(entity_type, []):
            if entity.get("id"):
                all_node_ids.add(entity["id"])
    
    # OCR validation rate
    ocr_validated = 0
    ocr_unvalidated = 0
    for entity_type in ["equipment", "instruments", "valves"]:
        for entity in nodes.get(entity_type, []):
            if entity.get("ocr_validated"):
                ocr_validated += 1
            else:
                ocr_unvalidated += 1
    
    total_entities = ocr_validated + ocr_unvalidated
    ocr_validation_rate = ocr_validated / total_entities if total_entities > 0 else 0
    
    # Lines with labels
    labeled_lines = 0
    propagated_labels = 0
    for line in process_lines + signal_lines:
        spec = line.get("specifications", {})
        if spec.get("line_number") or line.get("tag"):
            labeled_lines += 1
        if line.get("label_propagated"):
            propagated_labels += 1
    
    total_lines = len(process_lines) + len(signal_lines)
    label_coverage = labeled_lines / total_lines if total_lines > 0 else 0
    
    # === EDGE-CENTRIC METRICS (arXiv:2411.13929) ===
    
    # Count edge endpoints
    total_edges = 0
    connected_endpoints = 0
    dangling_endpoints = 0
    boundary_endpoints = 0
    seam_breaks = 0
    
    for line in process_lines + signal_lines:
        total_edges += 1
        
        # Check source connection
        source_id = line.get("source", {}).get("node_id")
        if source_id:
            if source_id in all_node_ids:
                connected_endpoints += 1
            else:
                dangling_endpoints += 1
        
        # Check target connection
        target_id = line.get("target", {}).get("node_id")
        if target_id:
            if target_id in all_node_ids:
                connected_endpoints += 1
            else:
                dangling_endpoints += 1
        
        # Check for boundary endpoints (seam issues)
        waypoints = line.get("route_waypoints", [])
        if waypoints:
            for wp in [waypoints[0], waypoints[-1]] if len(waypoints) >= 2 else waypoints:
                if is_near_boundary(wp):
                    boundary_endpoints += 1
        
        # Check for marked seam breaks
        if line.get("_seam_break"):
            seam_breaks += 1
    
    total_endpoints = (connected_endpoints + dangling_endpoints) or 1
    edge_connection_rate = connected_endpoints / total_endpoints
    dangling_ref_rate = dangling_endpoints / total_endpoints
    seam_break_rate = seam_breaks / total_lines if total_lines > 0 else 0
    
    # Tag grounding rate (how many tags are OCR-verified)
    grounded_tags = 0
    ungrounded_tags = 0
    for entity_type in ["equipment", "instruments", "valves"]:
        for entity in nodes.get(entity_type, []):
            if entity.get("tag"):
                if entity.get("ocr_validated") or entity.get("tag_verified", True):
                    grounded_tags += 1
                else:
                    ungrounded_tags += 1
    
    total_tagged = grounded_tags + ungrounded_tags
    tag_grounding_rate = grounded_tags / total_tagged if total_tagged > 0 else 1.0
    
    return {
        # Entity counts
        "total_equipment": equipment_count,
        "total_instruments": instrument_count,
        "total_valves": valve_count,
        "total_process_lines": len(process_lines),
        "total_signal_lines": len(signal_lines),
        "total_entities": total_entities,
        "total_lines": total_lines,
        # OCR metrics
        "ocr_validated": ocr_validated,
        "ocr_unvalidated": ocr_unvalidated,
        "ocr_validation_rate": round(ocr_validation_rate, 3),
        # Label metrics
        "labeled_lines": labeled_lines,
        "propagated_labels": propagated_labels,
        "label_coverage": round(label_coverage, 3),
        # Edge-centric metrics (NEW)
        "connected_endpoints": connected_endpoints,
        "dangling_endpoints": dangling_endpoints,
        "edge_connection_rate": round(edge_connection_rate, 3),
        "dangling_ref_rate": round(dangling_ref_rate, 3),
        "boundary_endpoints": boundary_endpoints,
        "seam_breaks": seam_breaks,
        "seam_break_rate": round(seam_break_rate, 3),
        # Tag grounding metrics (NEW)
        "grounded_tags": grounded_tags,
        "ungrounded_tags": ungrounded_tags,
        "tag_grounding_rate": round(tag_grounding_rate, 3),
        # Validation metrics
        "unresolved_references": len(validation.get("unresolved_references", [])),
        "orphan_nodes": len(validation.get("orphaned_nodes", [])),
        "incomplete_loops": len(validation.get("incomplete_loops", [])),
        "orthogonality_warnings": len(validation.get("orthogonality_warnings", [])),
        "is_valid": validation.get("is_valid", False)
    }


def is_near_boundary(point: list, threshold: int = 20) -> bool:
    """Check if a point is near the tile boundary (0 or 1000)."""
    if not point or len(point) < 2:
        return False
    x, y = point[0], point[1]
    return (x <= threshold or x >= 1000 - threshold or 
            y <= threshold or y >= 1000 - threshold)


def compare_pipelines(metrics_a: dict, metrics_b: dict, metrics_c: dict = None) -> dict:
    """
    Compare metrics between pipeline configurations.
    """
    comparison = {
        "baseline_a": metrics_a,
        "with_cv_b": metrics_b,
        "improvements_b_vs_a": {}
    }
    
    # Calculate improvements B vs A
    for key in metrics_a:
        if isinstance(metrics_a[key], (int, float)) and key != "is_valid":
            if "rate" in key or "coverage" in key:
                # Higher is better
                diff = metrics_b[key] - metrics_a[key]
            elif "unresolved" in key or "orphan" in key or "warning" in key or "unvalidated" in key:
                # Lower is better (show as negative = improvement)
                diff = metrics_a[key] - metrics_b[key]
            else:
                # More entities = more detection
                diff = metrics_b[key] - metrics_a[key]
            
            comparison["improvements_b_vs_a"][key] = diff
    
    if metrics_c:
        comparison["with_bfs_c"] = metrics_c
        comparison["improvements_c_vs_b"] = {}
        
        for key in metrics_b:
            if isinstance(metrics_b[key], (int, float)) and key != "is_valid":
                if "rate" in key or "coverage" in key:
                    diff = metrics_c[key] - metrics_b[key]
                elif "unresolved" in key or "orphan" in key or "warning" in key:
                    diff = metrics_b[key] - metrics_c[key]
                else:
                    diff = metrics_c[key] - metrics_b[key]
                
                comparison["improvements_c_vs_b"][key] = diff
    
    return comparison


def print_metrics(metrics: dict, label: str):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    print(f"  Entities:")
    print(f"    Equipment:   {metrics['total_equipment']}")
    print(f"    Instruments: {metrics['total_instruments']}")
    print(f"    Valves:      {metrics['total_valves']}")
    print(f"    Lines:       {metrics['total_lines']}")
    print(f"  OCR Validation:")
    print(f"    Validated:   {metrics['ocr_validated']} ({metrics['ocr_validation_rate']*100:.1f}%)")
    print(f"    Unvalidated: {metrics['ocr_unvalidated']}")
    print(f"  Label Coverage:")
    print(f"    Labeled:     {metrics['labeled_lines']} ({metrics['label_coverage']*100:.1f}%)")
    print(f"    Propagated:  {metrics['propagated_labels']}")
    print(f"  Validation Issues:")
    print(f"    Unresolved refs:     {metrics['unresolved_references']}")
    print(f"    Orphan nodes:        {metrics['orphan_nodes']}")
    print(f"    Incomplete loops:    {metrics['incomplete_loops']}")
    print(f"    Orthogonality warns: {metrics['orthogonality_warnings']}")
    print(f"  Overall: {'[VALID]' if metrics['is_valid'] else '[INVALID]'}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare P&ID extraction pipelines"
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="One or more merged JSON files to evaluate"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Evaluate a single file (no comparison)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output comparison results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load and evaluate each file
    all_metrics = []
    labels = ["Pipeline A (Baseline)", "Pipeline B (+CV)", "Pipeline C (+BFS)"]
    
    for i, json_path in enumerate(args.json_files[:3]):
        path = Path(json_path)
        if not path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)
        
        with open(path) as f:
            merged = json.load(f)
        
        metrics = evaluate_extraction(merged)
        all_metrics.append(metrics)
        
        label = labels[i] if i < len(labels) else f"Pipeline {i+1}"
        print_metrics(metrics, f"{label}: {path.name}")
    
    # Compare if multiple files
    if len(all_metrics) >= 2 and not args.single:
        print(f"\n{'='*50}")
        print(" COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        comparison = compare_pipelines(
            all_metrics[0],
            all_metrics[1],
            all_metrics[2] if len(all_metrics) > 2 else None
        )
        
        print("\n  B vs A Improvements:")
        for key, diff in comparison["improvements_b_vs_a"].items():
            symbol = "+" if diff > 0 else ""
            if diff != 0:
                print(f"    {key}: {symbol}{diff:.3f}" if isinstance(diff, float) else f"    {key}: {symbol}{diff}")
        
        if "improvements_c_vs_b" in comparison:
            print("\n  C vs B Improvements:")
            for key, diff in comparison["improvements_c_vs_b"].items():
                symbol = "+" if diff > 0 else ""
                if diff != 0:
                    print(f"    {key}: {symbol}{diff:.3f}" if isinstance(diff, float) else f"    {key}: {symbol}{diff}")
        
        # Save comparison if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
