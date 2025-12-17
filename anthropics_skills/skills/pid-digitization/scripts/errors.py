#!/usr/bin/env python3
"""
P&ID Extraction Error Handling Module

Provides structured error output for P&ID digitization scripts.
All errors include error codes, descriptions, and optional region/context data.

Error Code Prefixes:
  - IMG_: Image processing errors
  - OCR_: OCR/text extraction errors
  - EXT_: Vision extraction errors
  - MRG_: Merge/validation errors
  - VAL_: Validation errors
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = "warning"     # Non-fatal, processing can continue
    ERROR = "error"         # Fatal for this item, but processing can continue
    CRITICAL = "critical"   # Fatal, processing must stop


class ErrorCode(Enum):
    """Structured error codes for P&ID extraction."""
    # Image processing errors (IMG_)
    IMG_NOT_FOUND = "IMG_001"
    IMG_INVALID_FORMAT = "IMG_002"
    IMG_TOO_LARGE = "IMG_003"
    IMG_SLICE_FAILED = "IMG_004"
    
    # OCR errors (OCR_)
    OCR_ENGINE_MISSING = "OCR_001"
    OCR_FAILED = "OCR_002"
    OCR_LOW_CONFIDENCE = "OCR_003"
    OCR_NO_TEXT_FOUND = "OCR_004"
    
    # Extraction errors (EXT_)
    EXT_UNREADABLE_REGION = "EXT_001"
    EXT_AMBIGUOUS_SYMBOL = "EXT_002"
    EXT_PARTIAL_TAG = "EXT_003"
    EXT_COORDINATE_MISMATCH = "EXT_004"
    EXT_VISION_TOOL_FAILED = "EXT_005"
    
    # Merge errors (MRG_)
    MRG_DUPLICATE_ID = "MRG_001"
    MRG_COORDINATE_TRANSFORM_FAILED = "MRG_002"
    MRG_NO_TILES_FOUND = "MRG_003"
    MRG_INVALID_JSON = "MRG_004"
    
    # Validation errors (VAL_)
    VAL_UNRESOLVED_REFERENCE = "VAL_001"
    VAL_ORPHANED_NODE = "VAL_002"
    VAL_INCOMPLETE_LOOP = "VAL_003"
    VAL_INVALID_BBOX = "VAL_004"
    VAL_MISSING_REQUIRED_FIELD = "VAL_005"


@dataclass
class ExtractionError:
    """Structured error for P&ID extraction."""
    code: str
    message: str
    severity: str = "error"
    region_bbox: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]
    source_file: Optional[str] = None
    source_tile: Optional[str] = None
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {
            "error": self.code,
            "message": self.message,
            "severity": self.severity
        }
        if self.region_bbox:
            result["region_bbox"] = self.region_bbox
        if self.source_file:
            result["source_file"] = self.source_file
        if self.source_tile:
            result["source_tile"] = self.source_tile
        if self.entity_id:
            result["entity_id"] = self.entity_id
        if self.entity_type:
            result["entity_type"] = self.entity_type
        if self.context:
            result["context"] = self.context
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class ErrorCollector:
    """Collects and manages errors during extraction."""
    
    def __init__(self):
        self.errors: List[ExtractionError] = []
        self._has_critical = False
    
    def add(self, error: ExtractionError):
        """Add an error to the collection."""
        self.errors.append(error)
        if error.severity == "critical":
            self._has_critical = True
    
    def add_warning(self, code: ErrorCode, message: str, **kwargs):
        """Add a warning."""
        self.add(ExtractionError(
            code=code.value,
            message=message,
            severity="warning",
            **kwargs
        ))
    
    def add_error(self, code: ErrorCode, message: str, **kwargs):
        """Add an error."""
        self.add(ExtractionError(
            code=code.value,
            message=message,
            severity="error",
            **kwargs
        ))
    
    def add_critical(self, code: ErrorCode, message: str, **kwargs):
        """Add a critical error."""
        self.add(ExtractionError(
            code=code.value,
            message=message,
            severity="critical",
            **kwargs
        ))
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def has_critical(self) -> bool:
        """Check if any critical errors were collected."""
        return self._has_critical
    
    def get_warnings(self) -> List[ExtractionError]:
        """Get all warnings."""
        return [e for e in self.errors if e.severity == "warning"]
    
    def get_errors(self) -> List[ExtractionError]:
        """Get all errors (excluding warnings)."""
        return [e for e in self.errors if e.severity in ("error", "critical")]
    
    def to_list(self) -> List[dict]:
        """Convert all errors to list of dicts."""
        return [e.to_dict() for e in self.errors]
    
    def to_json(self) -> str:
        """Convert all errors to JSON string."""
        return json.dumps(self.to_list(), indent=2)
    
    def summary(self) -> dict:
        """Get error summary counts."""
        return {
            "total": len(self.errors),
            "warnings": len([e for e in self.errors if e.severity == "warning"]),
            "errors": len([e for e in self.errors if e.severity == "error"]),
            "critical": len([e for e in self.errors if e.severity == "critical"])
        }
    
    def clear(self):
        """Clear all errors."""
        self.errors = []
        self._has_critical = False


# Convenience functions for creating common errors
def unreadable_region(bbox: List[int], source_tile: str = None) -> ExtractionError:
    """Create an unreadable region error."""
    return ExtractionError(
        code=ErrorCode.EXT_UNREADABLE_REGION.value,
        message="Region is unreadable or too degraded for extraction",
        severity="warning",
        region_bbox=bbox,
        source_tile=source_tile
    )


def ambiguous_symbol(entity_id: str, bbox: List[int], note: str = None) -> ExtractionError:
    """Create an ambiguous symbol error."""
    return ExtractionError(
        code=ErrorCode.EXT_AMBIGUOUS_SYMBOL.value,
        message=f"Symbol is ambiguous: {note or 'multiple interpretations possible'}",
        severity="warning",
        entity_id=entity_id,
        region_bbox=bbox,
        context={"ambiguity_note": note} if note else None
    )


def partial_tag(entity_id: str, partial_text: str, bbox: List[int] = None) -> ExtractionError:
    """Create a partial tag detection error."""
    return ExtractionError(
        code=ErrorCode.EXT_PARTIAL_TAG.value,
        message=f"Tag partially visible: '{partial_text}'",
        severity="warning",
        entity_id=entity_id,
        region_bbox=bbox,
        context={"partial_text": partial_text, "tag_complete": False}
    )


def unresolved_reference(source_id: str, target_id: str, field: str) -> ExtractionError:
    """Create an unresolved reference validation error."""
    return ExtractionError(
        code=ErrorCode.VAL_UNRESOLVED_REFERENCE.value,
        message=f"Reference to unknown entity: {source_id}.{field} -> {target_id}",
        severity="error",
        entity_id=source_id,
        context={"missing_reference": target_id, "field": field}
    )


def orphaned_node(entity_id: str, entity_type: str) -> ExtractionError:
    """Create an orphaned node validation error."""
    return ExtractionError(
        code=ErrorCode.VAL_ORPHANED_NODE.value,
        message=f"Equipment has no connections: {entity_id}",
        severity="warning",
        entity_id=entity_id,
        entity_type=entity_type
    )


# Global error collector instance
_global_collector = ErrorCollector()


def get_global_collector() -> ErrorCollector:
    """Get the global error collector instance."""
    return _global_collector


def reset_global_collector():
    """Reset the global error collector."""
    _global_collector.clear()
