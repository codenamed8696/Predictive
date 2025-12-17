# P&ID Extraction Guide

Detailed phase-by-phase instructions for P&ID digitization.

---

## Phase 0: Legend & Metadata Extraction

**System Role**: Expert Industrial Automation Engineer specializing in ISA-5.1 standards.
**Purpose**: Establish the diagram's symbology vocabulary before component extraction.

### Instructions

**Prompt Template**:
```text
You are an expert Industrial Automation Engineer. Analyze the attached P&ID to extract metadata and legend definitions.

1. **Scan Document Periphery**: Check corners/edges for tables labeled "Legend", "Symbols", "Abbreviations", "Line Types", "Notes".
2. **Extract Title Block**: Locate (typically bottom-right) and capture Drawing Number, Revision, Date, Project Name, Unit/Area.
3. **Transcribe Symbol Definitions**: Record symbol description and meaning.
4. **Identify Line Type Key**: Note line styles and service types.

Return JSON matching the Phase 0 schema.
```

### Extraction Steps
1. **Scan Document Periphery**: Check all four corners and edges for tables labeled:
   - "Legend", "Symbols", "Abbreviations", "Utility Keys", "Line Types", "Notes"
2. **Extract Title Block**: Locate (typically bottom-right) and capture:
   - Drawing Number, Revision, Date, Project Name, Unit/Area
3. **Transcribe Symbol Definitions**: For each legend entry, record the symbol image description and meaning
4. **Identify Line Type Key**: Note all line styles and their corresponding service types

---

## Phase 1: Major Equipment Extraction

**Purpose**: Identify all primary process equipment forming the backbone of the P&ID.

### Detection Rules (In-Context)
| Equipment Class | Visual Signature | Typical Tag Pattern |
|----------------|------------------|---------------------|
| Column/Tower | Tall vertical cylinder with internal trays | T-XXX, C-XXX |
| Vessel | Horizontal/vertical cylinder with heads | V-XXX, D-XXX |
| Tank | Large rectangular/cylindrical storage | TK-XXX |
| Pump | Circle with discharge arrow | P-XXX |
| Compressor | Circle with zigzag line | K-XXX, C-XXX |
| Heat Exchanger | Tube bundle or plate symbol | E-XXX, H-XXX |

### Instructions

**Prompt Template**:
```text
Extract Phase 1 (Equipment) from this P&ID tile.
Refer to the "Detection Rules" table above for visual signatures.

RULES:
1. **Tag Pattern**: Use format `[A-Z]-\d{2,4}[A-Z]?` (e.g., T-101, P-102A).
2. **Anchor Points**: Define connection nozzles (inlet/outlet) based on visible pipe connections.
3. **Coordinates**: Return [ymin, xmin, ymax, xmax] in normalized 0-1000 space.

Return JSON matching the Phase 1 schema.
```

### Extraction Steps
1. Identify all large geometric shapes distinct from piping
2. Extract the alphanumeric tag adjacent to each shape
3. Classify based on symbol shape following ISA standards
4. Calculate normalized bounding box
5. Define **anchor points** (connection nozzles) based on visible pipe connections

### Anchor Point Convention

```
Relative positions: top_center, bottom_center, left_center, right_center
Nozzle names: inlet, outlet, feed, overhead, bottoms, suction, discharge
Format: {"anchor_name": {"x": int, "y": int, "position": "string"}}
```

---

## Phase 2: Instrumentation & Control Extraction

**Purpose**: Extract the measurement and control layer—instrument bubbles and control loops.

### Detection Rules (ISA-5.1)
- **First Letter**: F=Flow, L=Level, P=Pressure, T=Temperature, A=Analysis
- **Succeeding**: I=Indicator, C=Controller, T=Transmitter, V=Valve
- **Mounting**: No Line=Field, Line=Control Room, Hexagon=PLC

### Instructions

**Prompt Template**:
```text
Extract Phase 2 (Instruments) from this P&ID tile.

RULES:
1. **Bubbles**: Locate circular/hexagonal bubbles containing text.
2. **Tag Parsing**: Split into Function (e.g., TIC) and Loop # (e.g., 101).
3. **Mounting**: Determine location from symbol modifier (line vs no line).
4. **Loops**: Group instruments sharing the same loop number.

Return JSON matching the Phase 2 schema.
```

### Extraction Steps
1. Locate circular/hexagonal bubbles containing text
2. Parse the Functional Identification (letters) and Loop Number (digits)
3. Determine mounting location from symbol modifier (line = control room)
4. Group instruments sharing the same loop number into logical control loops
5. Trace signal lines (dashed) connecting instruments within a loop

---

## Phase 3: Valve Extraction

**Purpose**: Identify all manual and control valves as critical inline components.

### Detection Rules
- **Gate/Globe**: Bowtie (two triangles)
- **Control**: Bowtie + Actuator (mushroom/diaphragm top)
- **Check**: Bowtie + Checkmark/Arrow
- **Ball**: Circle/Ball symbol inside line

### Instructions

**Prompt Template**:
```text
Extract Phase 3 (Valves) from this P&ID tile.

RULES:
1. **Classification**: Identify valve type from shape (see rules above).
2. **Actuators**: Check for diaphragm (curved top), motor (M), or solenoid (diagonal).
3. **Tags**: Extract tag (e.g., V-101) if present; manual valves may be untagged.
4. **Fail State**: Look for "FC" (Fail Closed) or "FO" (Fail Open).

Return JSON matching the Phase 3 schema.
```

### Extraction Steps
1. Identify valve symbols along piping routes
2. Determine valve type from symbol shape
3. Identify actuator type if present
4. Extract tag if labeled (many manual valves may be untagged)
5. Note fail position if indicated (FC = Fail Closed, FO = Fail Open)

---

## Phase 4: Piping Topology & Process Flow

**Purpose**: Trace all process lines establishing equipment connectivity and material flow.

### Core Invariant: Orthogonality
- **Constraint**: P&ID pipes NEVER travel diagonally.
- **Routing**: All paths MUST use right-angle turns.
- **Waypoints**: Trace path as `[[x1,y1], [x2,y2], ...]` corner points.

### Instructions

**Prompt Template**:
```text
Extract Phase 4 (Piping) from this P&ID tile.

RULES:
1. **Trace**: Follow lines from Source Node to Target Node.
2. **Orthogonality**: Use waypoints for ALL right-angle turns. No diagonal lines.
3. **Direction**: Use flow arrows or process logic (e.g., pump discharge).
4. **Specs**: Extract size/material text parallel to the line.

Return JSON matching the Phase 4 schema.
```

### Extraction Steps
1. Trace each line from source to destination equipment
2. Identify direction using:
   - Flow arrows on the line
   - Pump discharge direction
   - Logical process sequence
3. Extract line specifications written parallel to the line:
   - Size (e.g., 4")
   - Schedule/Rating (e.g., Sch 40)
   - Material Code (e.g., CS = Carbon Steel)
   - Insulation code if present
4. **CRITICAL**: Provide `route_waypoints` array tracing the exact path with corners
5. Note all inline components (valves, instruments) the line passes through

---

## Phase 5: Process Logic Extraction

**Purpose**: Capture semantic process relationships for Digital Twin construction.

### Instructions

**Prompt Template**:
```text
Extract Phase 5 (Process Logic) from this P&ID tile.

RULES:
1. **Flow Sequences**: Trace the primary material path through equipment chains.
2. **Control Logic**: Link Sensors -> Controllers -> Final Elements (Valves).
3. **Interlocks**: Identify safety shutdowns (e.g., "High Level Trip").
4. **Setpoints**: Extract specific process values (e.g., "150 PSI").

Return JSON matching the Phase 5 schema.
```

### Extraction Steps
1. Trace material flow sequences through equipment chains
2. Identify control loop relationships (sensor → controller → final element)
3. Extract interlock logic and safety shutdown triggers
4. Note stream splits and merges
5. Capture setpoints and alarm limits where visible

## Phase 6: Validation & Cross-Reference

**Purpose**: Ensure data integrity by validating extracted information.

### Validation Checks
1. **Tag Uniqueness**: Every ID must be unique across all phases
2. **Reference Integrity**: All referenced IDs in `source`, `target`, `inline_components` must exist
3. **Connectivity Validation**: Each equipment anchor used should have a corresponding line
4. **Loop Completeness**: Control loops should have at least a sensor and final element
5. **Orphan Detection**: Flag equipment with no connecting lines

### Instructions

**Prompt Template**:
```text
Perform Phase 6 (Validation) on the extracted data.

RULES:
1. **Cross-Check**: Verify all node IDs (Equipment, Valves, Instruments) exist in the extraction list.
2. **Connectivity**: Flag any equipment with 0 connections.
3. **Loops**: Ensure every Control Loop has at least 1 sensor and 1 output.
4. **Report**: detailed `unresolved_references` and `orphaned_nodes`.

Return JSON matching the Phase 6 schema.
```

### Extraction Steps
1. Cross-reference all node IDs between phases
2. Validate all references resolve correctly
3. Identify any gaps or inconsistencies
4. Report validation issues without modifying data

---

## Processing Strategies

### Iterative Refinement Strategy

For complex diagrams, use multi-pass extraction:

1. **First Pass**: Run Phase 1 (Equipment) to establish anchor nodes
2. **Second Pass**: Run Phases 2-3 with equipment list as context
3. **Third Pass**: Run Phase 4 with complete node inventory
4. **Validate**: Any reference to unknown nodes triggers re-analysis of that region

### Tiling Strategy (for high-resolution images)

1. **Divide**: Use `slice_image.py` to split based on resolution
2. **Process**: Run Phases 1-4 on each tile independently
3. **Overlap**: Use 10% overlap between tiles to catch boundary elements
4. **Deduplicate**: Use `merge_json.py` to resolve duplicates by:
   - Tag matching (same tag = same element)
   - Spatial proximity (bbox overlap > 80% = duplicate)
5. **Reconnect**: Lines cut by tile boundaries are joined

---

## Usage Examples

### Single-Phase Invocation

```
Extract Phase 1 (Equipment) from the attached P&ID image.
Follow the CORE INVARIANTS and Phase 1 instructions exactly.
Return only the Phase 1 JSON schema.
```

### Full Extraction Invocation

```
Perform complete P&ID extraction on the attached image.
Execute Phases 0-6 sequentially.
Return the FINAL CONSOLIDATED OUTPUT schema.
```

### With MCP Tools

```
1. Use MCP vision tool to analyze this P&ID tile
2. Extract all equipment following Phase 1 instructions
3. For each equipment item, identify:
   - Tag (alphanumeric label)
   - Class (vessel, pump, etc.)
   - Bounding box in normalized coordinates
   - Connection anchors
4. Return structured JSON per complete-schema.md
```
