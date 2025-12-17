# Complete JSON Schema

Consolidated output schema for P&ID extraction (Phases 0-6).

> [!IMPORTANT]
> **Schema Version**: 1.1.1 (December 2024)
> **Coordinate Format**: ALL `bbox` fields use `[ymin, xmin, ymax, xmax]` in **normalized 0-1000 space**.
> Top-Left = `[0, 0]`, Bottom-Right = `[1000, 1000]`. Never use raw pixel coordinates.

## Final Output Structure

```json
{
  "schema_version": "1.1.1",
  "extraction_timestamp": "ISO8601",
  "source_image": "filename",
  "metadata": {},
  "legend": {},
  "nodes": {
    "equipment": [],
    "instruments": [],
    "valves": []
  },
  "edges": {
    "process_lines": [],
    "signal_lines": []
  },
  "process_logic": {
    "flow_sequences": [],
    "control_relationships": [],
    "interlocks": []
  },
  "utilities": [],
  "validation": {}
}
```

---

## Phase 0: Metadata & Legend

```json
{
  "metadata": {
    "drawing_number": "string | null",
    "revision": "string | null",
    "date": "string | null",
    "project": "string | null",
    "unit_area": "string | null",
    "title": "string | null",
    "notes": ["string"]
  },
  "legend": {
    "symbols": [
      {"symbol_id": "string", "description": "string", "category": "Equipment | Valve | Instrument | Fitting | Other"}
    ],
    "line_types": [
      {
        "style": "solid-thick | solid-medium | dashed | dash-dot",
        "service": "Process | Signal | Steam | Utility | Other",
        "color": "string | null",
        "size": "string | null",
        "description": "string | null"
      }
    ],
    "abbreviations": [
      {"code": "string", "meaning": "string"}
    ]
  }
}
```

---

## Phase 1: Equipment

> **ID Format**: Use `EQ-XXX` pattern (e.g., `EQ-001`, `EQ-002`).

```json
{
  "equipment": [
    {
      "id": "EQ-001",
      "tag": "T-101",
      "class": "Column | Vessel | Tank | Pump | Compressor | HeatExchanger | Reactor | Filter | Other",
      "subtype": "string | null",
      "bbox": [100, 200, 300, 400],
      "anchors": {
        "anchor_name": {"x": 250, "y": 100, "position": "top_center | bottom_center | left_center | right_center"}
      },
      "connections": {
        "inlet_from": ["VLV-001"],
        "outlet_to": ["EQ-002"]
      },
      "interlocks": ["IL-001"],
      "description": "string | null",
      "confidence": 0.85
    }
  ]
}
```

---

## Phase 2: Instruments

```json
{
  "instruments": [
    {
      "id": "string",
      "tag": "string",
      "function_letters": "string",
      "loop_number": "string",
      "full_function": "string",
      "mounting": "Field | ControlRoom | PLC | SharedDisplay",
      "bbox": [0, 0, 1000, 1000],
      "connected_to": "equipment_id | valve_id | line_id",
      "confidence": 0.0
    }
  ],
  "control_loops": [
    {
      "loop_id": "string",
      "description": "string | null",
      "instruments": ["instrument_id"],
      "control_strategy": "Feedback | Cascade | Ratio | Feedforward | null"
    }
  ]
}
```

---

## Phase 3: Valves

> **ID Format**: Use `VLV-XXX` pattern (e.g., `VLV-001`, `VLV-002`).

```json
{
  "valves": [
    {
      "id": "VLV-001",
      "tag": "V-101",
      "class": "Gate | Globe | Ball | Butterfly | Check | Control | Relief | ThreeWay | Other",
      "actuator": "Manual | Diaphragm | Motor | Solenoid | Handwheel | null",
      "fail_position": "FC | FO | null",
      "bbox": [450, 300, 480, 330],
      "upstream": "EQ-001",
      "downstream": "EQ-002",
      "controlled_by": "INST-003",
      "confidence": 0.9
    }
  ]
}
```

---

## Phase 4: Piping

```json
{
  "process_lines": [
    {
      "id": "string",
      "source": {"node_id": "string", "anchor": "string | null", "type": "Equipment | Valve | Utility | OffPage"},
      "target": {"node_id": "string", "anchor": "string | null", "type": "Equipment | Valve | Utility | OffPage"},
      "line_type": "Process_Major | Process_Minor | Signal | Steam | CoolingWater | Nitrogen | Other",
      "specifications": {
        "size_inches": 0,
        "schedule": "string | null",
        "material": "string | null",
        "insulation": "string | null",
        "full_spec_string": "string | null"
      },
      "contents": "string | null",
      "flow_direction": "Forward | Reverse | Bidirectional",
      "route_waypoints": [[0,0], [0,0]],
      "inline_components": ["valve_id", "instrument_id"],
      "confidence": 0.0
    }
  ],
  "utilities": [
    {
      "id": "string",
      "type": "Steam | CoolingWater | Nitrogen | Air | DrainToGrade | Other",
      "position": {"x": 0, "y": 0},
      "connected_to": "equipment_id | line_id"
    }
  ],
  "signal_lines": [
    {
      "id": "SIG-001",
      "source": {"instrument_id": "INST-001", "type": "Sensor | Transmitter"},
      "target": {"instrument_id": "INST-002", "type": "Controller | Valve | Indicator"},
      "signal_type": "Analog | Digital | Pneumatic | Hydraulic | Electrical",
      "line_style": "Dashed | Dash-Dot",
      "route_waypoints": [[100, 200], [100, 300], [200, 300]],
      "loop_id": "LOOP-101",
      "confidence": 0.85
    }
  ]
}
```

---

## Phase 5: Process Logic

```json
{
  "process_logic": {
    "flow_sequences": [
      {
        "sequence_id": "string",
        "description": "string",
        "nodes": ["equipment_id_1", "equipment_id_2"],
        "stream_type": "Main | Recycle | Utility | Bypass"
      }
    ],
    "control_relationships": [
      {
        "loop_id": "string",
        "sensor": "instrument_id",
        "controller": "instrument_id",
        "final_element": "valve_id",
        "control_action": "Direct | Reverse",
        "setpoint": "string | null",
        "alarm_high": "string | null",
        "alarm_low": "string | null"
      }
    ],
    "interlocks": [
      {
        "interlock_id": "string",
        "trigger_instrument": "instrument_id",
        "trigger_condition": "string",
        "action": "string",
        "affected_equipment": ["equipment_id", "valve_id"],
        "sil_rating": "SIL1 | SIL2 | SIL3 | null"
      }
    ]
  }
}
```

---

## Phase 6: Validation

```json
{
  "validation": {
    "total_equipment": 0,
    "total_instruments": 0,
    "total_valves": 0,
    "total_lines": 0,
    "orphaned_nodes": ["node_id"],
    "unresolved_references": [
      {"source_phase": 0, "source_id": "string", "missing_reference": "string", "field": "string"}
    ],
    "incomplete_loops": ["loop_id"],
    "warnings": ["string"],
    "is_valid": true
  }
}
```
