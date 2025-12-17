# Agentic Predictive Maintenance System (P&ID Digitization)

## 1. Project Overview
This project is an **Agentic Predictive Maintenance Platform** designed to ingest static P&ID (Piping & Instrumentation Diagram) images and convert them into a live, reasoning "Digital Twin" of the plant.

The core goal is to enable **100% accurate digitization** of dense engineering diagrams into a structured JSON graph (`plant_topology.json`).

## 2. Methodology: Vision-Language Model (VLM) Extraction

We pivoted from a traditional OCR/Detection pipeline to a **Generative AI approach** using **Qwen2.5-VL-3B-Instruct**.

### Why this approach?
*   **Contextual Understanding:** Unlike OCR which reads text blindly, a VLM understands that a box with "P-101" inside it is a *Pump*, and a line pointing to it is an *Inlet*.
*   **JSON Generation:** The model is prompted to output strictly formatted JSON, reducing the need for complex post-processing heuristics.
*   **Local Execution:** The 3B parameter model is lightweight enough to run on a standard laptop CPU/GPU without cloud APIs.

### Model Architecture
*   **Base Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
*   **Fine-Tuning (Optional/Experimental):** We explored using a LoRA adapter (`zackriya/diagram2graph-adapters`) specifically fine-tuned for diagrams. Currently, the system uses the **Base Model** as it demonstrated zero-shot performance on our P&ID tiles without the complexity of adapter loading.

## 3. Codebase Guide for Developers

### Core Scripts (`src/digitization/`)

#### 1. `qwen_inference.py` (The Engine)
*   **Purpose:** Runs the VLM on a single image tile.
*   **Key Functions:**
    *   `run_inference(image_path)`: Loads the model (if not loaded), processes the image, and generates JSON.
    *   **Prompt Engineering:** Uses a specific `SYSTEM_MESSAGE` to constrain the model to output *only* JSON with `nodes` and `edges`.
*   **Configuration:**
    *   `MODEL_ID`: Set to `Qwen/Qwen2.5-VL-3B-Instruct`.
    *   `device_map="auto"`: Automatically offloads to CPU/Disk if VRAM is insufficient.
    *   `torch_dtype`: Forces `float32` for stable CPU execution.

#### 2. `tile_image.py` (Preprocessing)
*   **Purpose:** Slices large P&ID Master images (e.g., 8000x6000 px) into smaller, manageable tiles (1024x1024 px).
*   **Why:** VLMs have resolution limits. Sending the whole 8K image results in loss of detail (small text becomes unreadable). Tiling preserves resolution.

#### 3. `json_to_mermaid.py` (Visualization)
*   **Purpose:** Converts the extracted JSON topology into Mermaid.js flowchart syntax.
*   **Usage:** Useful for quickly debugging if the extracted graph makes sense visually.

### Directory Structure
```text
D:\Predictive\
├── data\                   # Data artifacts
│   ├── topology\           # JSON outputs
├── src\                    # Source code
│   ├── digitization\       # Image processing & AI logic
│   │   ├── qwen_inference.py
│   │   ├── tile_image.py
├── tiles_landscape_fixed\  # Generated image tiles (Input for AI)
└── offload\                # Temp storage for model offloading (created by accelerate)
```

## 4. Setup & Installation

### Environment
We use a virtual environment `.venv` to manage dependencies and avoid conflicts with global system packages.

### Dependencies
Critical libraries required for Qwen2.5-VL:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate peft qwen-vl-utils
```
*Note: `qwen-vl-utils` handles the complex video/image preprocessing specific to Qwen models.*

### 4.2 Running the Extraction (Local Execution)

The system is designed to run **entirely on your local machine**.

1.  **First Run (Download):** The first time you run the script, it will automatically download the `Qwen2.5-VL-3B-Instruct` model (~7GB) from HuggingFace to your local cache (`~/.cache/huggingface`). The `diagram2graph` adapter is expected in `local_adapter/` (or set `DIAGRAM2GRAPH_ADAPTER` to a different path/repo).
2.  **Subsequent Runs (Offline):** All future runs will use the cached local model. You do not need an internet connection.

**Command (Adapter Pipeline):**
To process all tiles and stitch them into a single `plant_topology.json`:
```bash
python src/digitization/adapter_pipeline.py --tiles_dir tiles_landscape_fixed --output_dir data/topology
```
Per-tile JSONs are written next to the stitched graph in `data/topology/`.

**Output:**
*   Console: Progress logs per tile.
*   Files: Per-tile JSONs in `data/topology/` plus a stitched `plant_topology.json`.

**Forcing Offline Mode:**
To ensure no internet is used, set the environment variable:
```powershell
$env:HF_HUB_OFFLINE=1
python src/digitization/qwen_inference.py ...
```

### 4.3 Visualizing the Output
To verify the JSON output visually, convert it to a Mermaid diagram (requires `json_to_mermaid.py` update for new schema):

```bash
# Generate Mermaid code
python src/digitization/json_to_mermaid.py tiles_landscape_fixed/tile_4_8.png.json
```
