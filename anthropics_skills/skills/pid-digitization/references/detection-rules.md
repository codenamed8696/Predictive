# Detection Rules

Tables for identifying P&ID components.

---

## Equipment Detection

| Equipment Class | Visual Signature | Tag Pattern |
|----------------|------------------|-------------|
| Column/Tower | Tall vertical cylinder with internal trays | T-XXX, C-XXX |
| Vessel | Horizontal/vertical cylinder with heads | V-XXX, D-XXX |
| Tank | Large rectangular/cylindrical storage | TK-XXX |
| Pump | Circle with discharge arrow | P-XXX |
| Compressor | Circle with zigzag line | K-XXX, C-XXX |
| Heat Exchanger | Tube bundle or plate symbol | E-XXX, H-XXX |
| Reactor | Vessel with agitator | R-XXX |
| Filter | Chamber with internal element | F-XXX |
| Conveyor/Feeder | Belt or screw mechanism | CV-XXX, FD-XXX |

---

## Instrument Detection (ISA-5.1)

### First Letter Codes (Measured Variable)

| Letter | Variable |
|--------|----------|
| A | Analysis |
| C | User Choice / Conductivity |
| D | Density / Differential |
| F | Flow |
| H | Hand (manual) |
| L | Level |
| P | Pressure |
| Q | Quantity / Event |
| S | Speed |
| T | Temperature |
| V | Vibration |
| W | Weight |

### Succeeding Letter Codes (Function)

| Letter | Function |
|--------|----------|
| A | Alarm |
| C | Controller |
| E | Element (Primary) |
| G | Glass / Gauge (sight) |
| H | High |
| I | Indicator |
| L | Low / Light |
| R | Recorder |
| S | Switch / Safety |
| T | Transmitter |
| V | Valve |
| Y | Relay / Compute |
| Z | Driver / Actuator |

### Symbol Shapes

| Shape | Mounting |
|-------|----------|
| Circle | Field-mounted |
| Circle + horizontal line | Control room |
| Hexagon | Computer/PLC |
| Diamond | Logic function |
| Square | Shared display |

---

## Valve Detection

| Symbol Type | Valve Class | Tag Pattern |
|-------------|-------------|-------------|
| Two triangles (bowtie) | Gate/Globe | V-XXX, HV-XXX |
| Bowtie + top actuator | Control Valve | FV-XXX, TV-XXX |
| Bowtie + checkmark | Check Valve | CV-XXX |
| Ball symbol | Ball Valve | BV-XXX |
| Butterfly shape | Butterfly | — |
| Three-way symbol | Three-Way | — |

### Actuator Types

| Symbol | Type |
|--------|------|
| Curved top | Diaphragm |
| M in circle | Motor |
| Diagonal line | Solenoid |
| Circle with cross | Handwheel |

---

## Line Type Detection

| Visual Style | Type | Service |
|--------------|------|---------|
| Thick solid | Process Major | Main flow |
| Medium solid | Process Minor | Secondary |
| Dashed | Signal | Control |
| Dash-dot | Steam | Utility |
| Dash-dot-dot | Other Utility | N2, Air, CW |
| Double line | Jacketed | Heated/cooled |

---

## Nomenclature Patterns

| Pattern | Type | Example |
|---------|------|---------|
| `[A-Z]-\d{2,4}[A-Z]?` | Equipment | T-101, P-102A |
| `[A-Z]{2,4}-\d{2,4}` | Instrument | TIC-101, PT-200 |
| `\d{1,2}"-[A-Z]{2,4}-\d+` | Line Number | 4"-PA-001 |
