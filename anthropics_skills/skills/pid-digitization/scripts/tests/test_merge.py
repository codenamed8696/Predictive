#!/usr/bin/env python3
"""
Unit tests for merge_json.py deduplication logic.

Run with: python -m pytest tests/test_merge.py -v
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from merge_json import (
    deduplicate_entities,
    calculate_bbox_overlap,
    reset_id_counters
)


class TestDeduplicateEntities:
    """Tests for the entity deduplication logic."""
    
    def setup_method(self):
        """Reset ID counters before each test."""
        reset_id_counters()
    
    def test_preserves_distinct_tags(self):
        """Entities with different tags but same location should NOT be merged."""
        entities = [
            {"id": "V1", "tag": "V-100", "bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"id": "V2", "tag": "V-101", "bbox": [100, 100, 200, 200], "confidence": 0.8},  # Same bbox!
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2, "Different tags should never be merged even with identical bbox"
    
    def test_merges_same_tags(self):
        """Entities with same tag and high overlap should be deduplicated."""
        entities = [
            {"id": "V1", "tag": "V-100", "bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"id": "V2", "tag": "V-100", "bbox": [100, 100, 200, 200], "confidence": 0.8},  # Same tag
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1, "Same tag with identical bbox should be merged"
        assert result[0]["confidence"] == 0.9, "Should keep higher confidence entity"
    
    def test_keeps_spatially_distinct_same_tag(self):
        """Same tag but spatially distinct should keep both (with rename)."""
        entities = [
            {"id": "V1", "tag": "V-100", "bbox": [0, 0, 100, 100], "confidence": 0.9},
            {"id": "V2", "tag": "V-100", "bbox": [500, 500, 600, 600], "confidence": 0.8},  # Far apart
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2, "Same tag but spatially distinct should keep both"
    
    def test_audit_log_records_merges(self):
        """Audit log should record what was merged/dropped."""
        entities = [
            {"id": "V1", "tag": "V-100", "bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"id": "V2", "tag": "V-100", "bbox": [100, 100, 200, 200], "confidence": 0.8},
        ]
        audit_log = []
        result = deduplicate_entities(entities, audit_log=audit_log)
        assert len(result) == 1
        assert len(audit_log) == 1
        assert audit_log[0]["action"] == "merged_duplicate"
    
    def test_high_overlap_threshold(self):
        """Should only merge at 95% overlap, not 80%."""
        # 80% overlap bbox
        entities = [
            {"id": "V1", "tag": "V-100", "bbox": [0, 0, 100, 100], "confidence": 0.9},
            {"id": "V2", "tag": "V-100", "bbox": [20, 20, 120, 120], "confidence": 0.8},  # ~64% overlap
        ]
        result = deduplicate_entities(entities, overlap_threshold=0.95)
        # With 95% threshold, 64% overlap should NOT trigger merge on spatial pass
        # But same tag will still group them
        assert len(result) == 2, "64% overlap should not trigger spatial merge with 0.95 threshold"
    
    def test_null_tags_spatial_merge(self):
        """Entities without tags should be merged based on spatial overlap only."""
        entities = [
            {"id": "V1", "tag": None, "bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"id": "V2", "tag": None, "bbox": [100, 100, 200, 200], "confidence": 0.8},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1, "Null tags with identical bbox should be merged"


class TestBboxOverlap:
    """Tests for bbox overlap calculation."""
    
    def test_identical_boxes(self):
        """Identical boxes should have 100% overlap."""
        bbox = [100, 100, 200, 200]
        overlap = calculate_bbox_overlap(bbox, bbox)
        assert overlap == 1.0
    
    def test_no_overlap(self):
        """Non-overlapping boxes should have 0% overlap."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0
    
    def test_partial_overlap(self):
        """Partially overlapping boxes should have proportional overlap."""
        bbox1 = [0, 0, 100, 100]  # 10000 area
        bbox2 = [50, 50, 150, 150]  # 10000 area, 2500 intersection
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert 0.2 < overlap < 0.3  # ~25% of smaller box


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
