# DOTA-VLM: Automated Annotation of Aerial Imagery Using Vision–Language Models

A Vision–Language pipeline for enriching, validating, and auto-annotating the DOTA aerial dataset with captions, attributes, and contextual metadata.

##  Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

##  Overview

DOTA-VLM is a semi-automatic annotation framework that enriches the original DOTA dataset using Vision–Language Models (VLMs) such as:

- **LLaVA** / LLaVA-NeXT
- **BLIP-2** / BLIP-3
- **Florence-2**
- **OWL-ViT**

This project adds fine-grained semantic metadata to DOTA, including:

-  Object attributes (size, orientation, activity, appearance)
-  Spatial relations between objects
-  Image captions and scene descriptions
-  Activity descriptions (parked, moving, docked)
-  Confidence & ambiguity indicators
-  Zero-shot class verification

The output is a **COCO-style JSON** with both original annotations and new VLM-generated metadata.

##  Key Features

- **Automatic object attribute generation** - Size, orientation, activity, appearance
- **Scene captioning** - Comprehensive aerial image descriptions
- **Spatial relationship extraction** - Positional and arrangement patterns
- **Zero-shot class verification** - Validate detections using VLM knowledge
- **Quality & uncertainty estimation** - Flag ambiguous or low-confidence labels
- **Plug-and-play detector support** - YOLO-OBB, Oriented RCNN, DiffusionDet, DINO-OBB
- **Configurable prompting system** - Customizable multi-modal annotation templates

##  Project Architecture

```
┌──────────────────────┐
│ DOTA Image (RGB)     │
└───────────┬──────────┘
            │
            ▼
┌────────────────────────────┐
│ Oriented Object Detector   │
│ (YOLO-OBB / DiffusionDet)  │
└───────────┬───────────────┘
            │ Rotated Boxes
            ▼
┌────────────────────────────┐
│ Object Crop Generator      │
└───────────┬───────────────┘
            │ Cropped Patches
            ▼
┌────────────────────────────┐
│ Vision–Language Model      │
│ (LLaVA / BLIP / Florence)  │
└───────────┬───────────────┘
            │ Text Metadata
            ▼
┌────────────────────────────┐
│ Annotation Merger + JSON   │
└────────────────────────────┘
```

##  Dataset

This project uses the **DOTA v1.0 / v1.5 / v2.0** dataset:

- Aerial images (large-size, multi-angle, multi-scale)
- 15 categories: airplane, ship, vehicle, tennis court, etc.
- Oriented bounding box format
- **Dataset link**: https://captain-whu.github.io/DOTA/dataset.html

##  Installation

### 1. Clone Repository

```bash
git clone https://github.com/HayLahav/DOTA-VLM.git
cd DOTA-VLM
```

### 2. Create Environment

```bash
conda create -n dota_vlm python=3.10 -y
conda activate dota_vlm
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- OpenCV
- Shapely
- Ultralytics (for YOLO-OBB)

### 4. Download Models (Optional)

```bash
# Download pre-trained YOLO-OBB model
mkdir -p checkpoints
# Place your trained YOLO-OBB model in checkpoints/

# LLaVA model will be downloaded automatically on first use
```

##  Quick Start

### Complete Pipeline

Run the entire pipeline with one command:

```bash
# 1. Run detection
python detection/run_detector.py \
    --input_dir data/DOTA/images \
    --output detections.json \
    --model_path checkpoints/yolo_obb.pt \
    --conf_threshold 0.3

# 2. Crop objects
python tools/crop_objects.py \
    --detections detections.json \
    --images_dir data/DOTA/images \
    --out_dir crops/

# 3. Generate VLM annotations
python vlm/generate_annotations.py \
    --crops_dir crops/ \
    --model llava \
    --output metadata.json

# 4. Merge annotations
python tools/merge_annotations.py \
    --dota_json detections.json \
    --vlm_json metadata.json \
    --output dota_vlm.json \
    --summary
```

##  Detailed Usage

### Step 1: Run Oriented Object Detector

Detect objects with oriented bounding boxes:

```bash
python detection/run_detector.py \
    --input_dir data/DOTA/images \
    --output detections.json \
    --model_path checkpoints/yolo_obb.pt \
    --detector yolo_obb \
    --conf_threshold 0.3
```

**Arguments:**
- `--input_dir`: Directory containing DOTA images
- `--output`: Output JSON file path
- `--model_path`: Path to detection model checkpoint
- `--detector`: Detector type (`yolo_obb`, `oriented_rcnn`)
- `--conf_threshold`: Confidence threshold (default: 0.3)

### Step 2: Crop Objects

Extract object patches from images:

```bash
python tools/crop_objects.py \
    --detections detections.json \
    --images_dir data/DOTA/images \
    --out_dir crops/ \
    --padding 10
```

**Arguments:**
- `--detections`: Path to detections JSON
- `--images_dir`: Directory with original images
- `--out_dir`: Output directory for crops
- `--padding`: Padding around crops in pixels (default: 10)

### Step 3: Generate VLM Annotations

Create rich annotations using Vision-Language Models:

```bash
python vlm/generate_annotations.py \
    --crops_dir crops/ \
    --model llava \
    --model_name llava-hf/llava-1.5-7b-hf \
    --output metadata.json \
    --annotations attributes verification uncertainty
```

**Arguments:**
- `--crops_dir`: Directory containing crops
- `--model`: VLM type (`llava`, `blip2`)
- `--model_name`: HuggingFace model identifier
- `--output`: Output JSON path
- `--annotations`: Types to generate (`attributes`, `verification`, `uncertainty`)

**Supported Models:**
- `llava-hf/llava-1.5-7b-hf` (recommended)
- `llava-hf/llava-1.5-13b-hf` (more accurate)
- `Salesforce/blip2-opt-2.7b`
- `Salesforce/blip2-flan-t5-xl`

### Step 4: Export DOTA-VLM JSON

Merge detections with VLM metadata in COCO format:

```bash
python tools/merge_annotations.py \
    --dota_json detections.json \
    --vlm_json metadata.json \
    --output dota_vlm.json \
    --include_unverified \
    --summary dota_vlm_summary.txt
```

**Arguments:**
- `--dota_json`: Detection results JSON
- `--vlm_json`: VLM annotations JSON
- `--output`: Output COCO-style JSON
- `--include_unverified`: Include objects without VLM data
- `--summary`: Generate summary report

##  Output Format

### DOTA-VLM COCO-Style JSON

```json
{
  "info": {
    "description": "DOTA-VLM Dataset",
    "version": "1.0"
  },
  "images": [
    {
      "id": 0,
      "file_name": "P0001.png",
      "width": 1024,
      "height": 1024
    }
  ],
  "categories": [
    {"id": 0, "name": "airplane"},
    {"id": 1, "name": "ship"}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [100, 200, 50, 30],
      "area": 1500,
      "oriented_bbox": {
        "center": [125, 215],
        "size": [50, 30],
        "angle": 45.0
      },
      "detection_score": 0.95,
      "vlm_metadata": {
        "attributes": "Large commercial airplane, white fuselage with blue tail, parked on tarmac, stationary",
        "class_verification": "YES - This is clearly an airplane based on wings, fuselage, and tail structure",
        "uncertainty": "High confidence - Object is clearly visible with no occlusions",
        "crop_size": [60, 50]
      }
    }
  ]
}
```

##  Configuration

### Custom Prompt Templates

Edit `vlm/generate_annotations.py` to customize prompts:

```python
class PromptTemplates:
    @staticmethod
    def object_attributes() -> str:
        return """Your custom prompt here"""
```

### Detector Configuration

For YOLO-OBB training, see [Ultralytics YOLO-OBB docs](https://docs.ultralytics.com/tasks/obb/)

##  Examples

### Example 1: Process Single Image

```python
from detection.run_detector import YOLOOBBDetector
from tools.crop_objects import crop_detections
from vlm.generate_annotations import LLaVAAnnotator

# Detect
detector = YOLOOBBDetector('checkpoints/yolo_obb.pt')
detections = detector.detect('image.jpg')

# Crop
crops = crop_detections('image.jpg', detections, 'crops/', 'img_001')

# Annotate
vlm = LLaVAAnnotator()
for crop in crops:
    metadata = vlm.generate_caption(crop['crop_path'], "Describe this object")
```

### Example 2: Batch Processing with Custom Settings

```bash
# High-confidence detections only
python detection/run_detector.py \
    --input_dir data/DOTA/val \
    --conf_threshold 0.5 \
    --output val_detections.json

# Large padding for context
python tools/crop_objects.py \
    --detections val_detections.json \
    --images_dir data/DOTA/val \
    --padding 20 \
    --out_dir val_crops/
```

##  Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller model or reduce batch size
python vlm/generate_annotations.py \
    --model_name llava-hf/llava-1.5-7b-hf  # Use 7B instead of 13B
```

### LLaVA Import Error

```bash
pip install git+https://github.com/haotian-liu/LLaVA.git
# Or use HuggingFace version
pip install transformers>=4.35.0
```

### Slow Processing

- Use GPU for VLM inference
- Process images in parallel (modify scripts)
- Use quantized models (add `load_in_8bit=True`)

##  Evaluation

Calculate annotation quality metrics:

```python
from tools.evaluate import calculate_metrics

metrics = calculate_metrics('dota_vlm.json', 'ground_truth.json')
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

##  Project Roadmap

- [x] YOLO-OBB detection support
- [x] LLaVA integration
- [x] BLIP-2 integration
- [ ] Florence-2 support
- [ ] Spatial relationship extraction
- [ ] Scene-level captioning
- [ ] Interactive annotation tool
- [ ] Uncertainty quantification
- [ ] Multi-GPU support

##  Citation

```bibtex
@article{dota-vlm2024,
  title={DOTA-VLM: Automated Annotation of Aerial Imagery Using Vision-Language Models},
  author= Hay Lahav,
  journal= Not published yet,
  year={2025}
}
```

##  License

This project is licensed under the MIT License. See `LICENSE` file for details.

##  Contributing

Contributions are welcome! Please submit pull requests or open issues.



