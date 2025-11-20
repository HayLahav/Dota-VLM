# DOTA-VLM Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOTA-VLM SYSTEM                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   INPUT LAYER                            │   │
│  │                                                           │   │
│  │  ┌──────────────┐      ┌──────────────┐                │   │
│  │  │ DOTA Images  │      │ Config YAML  │                │   │
│  │  │  (Aerial)    │      │  (Settings)  │                │   │
│  │  └──────┬───────┘      └──────┬───────┘                │   │
│  └─────────┼─────────────────────┼────────────────────────┘   │
│            │                     │                              │
│  ┌─────────▼─────────────────────▼────────────────────────┐   │
│  │              DETECTION MODULE                           │   │
│  │                                                           │   │
│  │  ┌────────────────────────────────────────┐            │   │
│  │  │  OrientedObjectDetector (Base)         │            │   │
│  │  │  ├─ YOLOOBBDetector                   │            │   │
│  │  │  ├─ OrientedRCNNDetector (future)     │            │   │
│  │  │  └─ DiffusionDetector (future)        │            │   │
│  │  └────────────────────────────────────────┘            │   │
│  │                                                           │   │
│  │  Input: RGB Images                                       │   │
│  │  Output: Detections JSON                                │   │
│  │    {                                                     │   │
│  │      "image_path": "...",                               │   │
│  │      "detections": [                                    │   │
│  │        {                                                 │   │
│  │          "bbox": [cx, cy, w, h, angle],                │   │
│  │          "score": 0.95,                                 │   │
│  │          "class_name": "airplane"                       │   │
│  │        }                                                 │   │
│  │      ]                                                   │   │
│  │    }                                                     │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │              CROPPING MODULE                             │   │
│  │                                                           │   │
│  │  Functions:                                              │   │
│  │  ├─ get_rotated_crop()                                  │   │
│  │  ├─ get_rotated_crop_from_corners()                    │   │
│  │  └─ crop_detections()                                   │   │
│  │                                                           │   │
│  │  Features:                                               │   │
│  │  ├─ Rotation handling                                   │   │
│  │  ├─ Perspective transform                               │   │
│  │  ├─ Padding control                                     │   │
│  │  └─ Size validation                                     │   │
│  │                                                           │   │
│  │  Input: Original images + Detections JSON               │   │
│  │  Output: Crop images + Crops metadata JSON              │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │             VLM ANNOTATION MODULE                        │   │
│  │                                                           │   │
│  │  ┌────────────────────────────────────────┐            │   │
│  │  │  VLMAnnotator (Base)                   │            │   │
│  │  │  ├─ LLaVAAnnotator                     │            │   │
│  │  │  ├─ BLIP2Annotator                     │            │   │
│  │  │  └─ Florence2Annotator (future)        │            │   │
│  │  └────────────────────────────────────────┘            │   │
│  │                                                           │   │
│  │  ┌────────────────────────────────────────┐            │   │
│  │  │  PromptTemplates                        │            │   │
│  │  │  ├─ object_attributes()                │            │   │
│  │  │  ├─ scene_caption()                    │            │   │
│  │  │  ├─ spatial_relations()                │            │   │
│  │  │  ├─ uncertainty_check()                │            │   │
│  │  │  └─ class_verification()               │            │   │
│  │  └────────────────────────────────────────┘            │   │
│  │                                                           │   │
│  │  Input: Crop images + Crop metadata                     │   │
│  │  Output: VLM annotations JSON                           │   │
│  │    {                                                     │   │
│  │      "attributes": "Large white airplane...",           │   │
│  │      "verification": "YES - clearly an airplane",       │   │
│  │      "uncertainty": "High confidence..."               │   │
│  │    }                                                     │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │              MERGING MODULE                              │   │
│  │                                                           │   │
│  │  ┌────────────────────────────────────────┐            │   │
│  │  │  COCOAnnotationBuilder                 │            │   │
│  │  │  ├─ add_categories()                   │            │   │
│  │  │  ├─ add_image()                        │            │   │
│  │  │  ├─ add_annotation()                   │            │   │
│  │  │  └─ build()                            │            │   │
│  │  └────────────────────────────────────────┘            │   │
│  │                                                           │   │
│  │  Combines:                                               │   │
│  │  ├─ Detection results (bboxes, scores)                 │   │
│  │  ├─ VLM annotations (text metadata)                    │   │
│  │  └─ Image metadata                                      │   │
│  │                                                           │   │
│  │  Input: Detections JSON + VLM JSON                      │   │
│  │  Output: COCO-style JSON with VLM metadata              │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │                 OUTPUT LAYER                             │   │
│  │                                                           │   │
│  │  ┌──────────────┐      ┌──────────────┐                │   │
│  │  │ DOTA-VLM     │      │   Summary    │                │   │
│  │  │ COCO JSON    │      │   Report     │                │   │
│  │  └──────────────┘      └──────────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Interactions

### 1. Detection Module → Cropping Module

```
Detection Output:
{
  "image_001.png": {
    "detections": [
      {
        "bbox": [512, 384, 100, 80, 45.0],
        "bbox_corners": [[...], [...], [...], [...]],
        "score": 0.95,
        "class_id": 0,
        "class_name": "airplane"
      }
    ]
  }
}
         ↓
Cropping Input: Reads detections + loads original images
         ↓
Cropping Output:
{
  "image_001.png": {
    "crops": [
      {
        "crop_path": "image_001.png/obj_0000_airplane.jpg",
        "object_id": 0,
        "bbox": [512, 384, 100, 80, 45.0],
        "class_name": "airplane",
        "score": 0.95,
        "crop_size": [110, 90]
      }
    ]
  }
}
```

### 2. Cropping Module → VLM Module

```
VLM Input: Reads crop images + metadata
         ↓
For each crop:
  1. Load image
  2. Generate prompt
  3. Run VLM inference
  4. Parse response
         ↓
VLM Output:
{
  "image_001.png": {
    "annotations": [
      {
        "object_id": 0,
        "crop_path": "...",
        "class_name": "airplane",
        "detection_score": 0.95,
        "attributes": "Large commercial airplane...",
        "class_verification": "YES - ...",
        "uncertainty": "High confidence..."
      }
    ]
  }
}
```

### 3. All Modules → Merging Module

```
Merge Process:
  1. Load detections JSON
  2. Load VLM annotations JSON
  3. Match by image_id + object_id
  4. Combine into COCO format
         ↓
Final Output (COCO JSON):
{
  "info": {...},
  "images": [
    {
      "id": 0,
      "file_name": "image_001.png",
      "width": 1024,
      "height": 1024
    }
  ],
  "categories": [
    {"id": 0, "name": "airplane"},
    ...
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [462, 344, 100, 80],
      "oriented_bbox": {
        "center": [512, 384],
        "size": [100, 80],
        "angle": 45.0
      },
      "detection_score": 0.95,
      "vlm_metadata": {
        "attributes": "...",
        "class_verification": "...",
        "uncertainty": "..."
      }
    }
  ]
}
```

## Data Flow Diagram

```
┌──────────┐
│  DOTA    │
│  Images  │
└────┬─────┘
     │
     │ RGB Images (1024×1024)
     │
     ▼
┌────────────────┐
│   Detection    │
│   (YOLO-OBB)   │
└────┬───────────┘
     │
     │ Oriented Bboxes
     │ [cx, cy, w, h, θ]
     │
     ▼
┌────────────────┐
│    Cropping    │
│  (Rotation)    │
└────┬───────────┘
     │
     │ Object Patches
     │ (Various sizes)
     │
     ▼
┌────────────────┐
│  VLM (LLaVA)   │
│  + Prompts     │
└────┬───────────┘
     │
     │ Text Descriptions
     │ (Attributes, etc.)
     │
     ▼
┌────────────────┐
│    Merging     │
│  (COCO Build)  │
└────┬───────────┘
     │
     │ COCO JSON
     │
     ▼
┌──────────┐
│  Output  │
│  + Stats │
└──────────┘
```

## Class Hierarchy

### Detection Module
```
OrientedObjectDetector (ABC)
│
├── YOLOOBBDetector
│   ├── load_model()
│   ├── detect() → List[Dict]
│   └── _convert_to_obb_format()
│
└── OrientedRCNNDetector (future)
    ├── load_model()
    └── detect()
```

### VLM Module
```
VLMAnnotator (ABC)
│
├── LLaVAAnnotator
│   ├── load_model()
│   ├── generate_caption() → str
│   └── _load_llava_alternative()
│
├── BLIP2Annotator
│   ├── load_model()
│   └── generate_caption() → str
│
└── Florence2Annotator (future)
```

### Merging Module
```
COCOAnnotationBuilder
│
├── __init__()
├── add_categories()
├── add_image()
├── add_annotation()
└── build() → Dict
```

## Configuration Flow

```
configs/config.yaml
        │
        ├─ data paths
        ├─ detection settings
        ├─ cropping parameters
        ├─ vlm configuration
        ├─ merging options
        └─ processing flags
        │
        ▼
   pipeline.py
        │
        ├─ Loads config
        ├─ Validates inputs
        ├─ Runs steps sequentially
        └─ Logs progress
        │
        ▼
Individual modules receive config subsets
```

## Error Handling Strategy

```
┌─────────────────────────────────────┐
│         Pipeline Execution          │
│                                     │
│  Try:                               │
│    ├─ Step 1: Detection            │
│    ├─ Step 2: Cropping             │
│    ├─ Step 3: VLM                  │
│    └─ Step 4: Merging              │
│                                     │
│  Except:                            │
│    ├─ Log error                    │
│    ├─ Save partial results         │
│    └─ Decision:                    │
│        ├─ skip_on_error=True       │
│        │    └─ Continue             │
│        └─ skip_on_error=False      │
│             └─ Abort                │
└─────────────────────────────────────┘
```

## Performance Optimization Points

### 1. Detection Bottlenecks
- **Solution**: Batch inference, GPU utilization
- **Impact**: 5-10x speedup

### 2. Cropping I/O
- **Solution**: Parallel processing, in-memory caching
- **Impact**: 2-3x speedup

### 3. VLM Inference
- **Solution**: Model quantization, batch generation
- **Impact**: 2-5x speedup, 50% memory reduction

### 4. Merging
- **Solution**: Streaming JSON, incremental building
- **Impact**: Constant memory usage

## Scalability Architecture

### Horizontal Scaling
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Worker  │  │ Worker  │  │ Worker  │
│   1     │  │   2     │  │   N     │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
           ┌──────▼──────┐
           │   Merger    │
           └─────────────┘
```

### Vertical Scaling
```
GPU Memory:
├─ Detection: 2-4 GB
├─ VLM: 8-16 GB
└─ Total: 10-20 GB

CPU Memory:
├─ Image loading: 1-2 GB
├─ Crop storage: 2-5 GB
└─ JSON building: 1-3 GB
```

## Extension Architecture

### Adding New VLM
```python
class NewVLMAnnotator(VLMAnnotator):
    def load_model(self):
        # Load your model
        pass
    
    def generate_caption(self, image_path, prompt):
        # Generate caption
        return caption
```

### Adding New Detector
```python
class NewDetector(OrientedObjectDetector):
    def load_model(self):
        # Load your model
        pass
    
    def detect(self, image_path):
        # Run detection
        return detections
```

## Testing Architecture

```
tests/
├─ unit/
│  ├─ test_detection.py
│  ├─ test_cropping.py
│  ├─ test_vlm.py
│  └─ test_merging.py
│
├─ integration/
│  ├─ test_pipeline.py
│  └─ test_end_to_end.py
│
└─ performance/
   ├─ benchmark_detection.py
   ├─ benchmark_vlm.py
   └─ profile_memory.py
```

## Logging Architecture

```
Logs:
├─ pipeline.log (main)
│  ├─ Timestamp
│  ├─ Step name
│  ├─ Status
│  └─ Metrics
│
├─ detection.log
│  └─ Per-image results
│
├─ vlm.log
│  └─ Generation stats
│
└─ errors.log
   └─ Error traces
```

---

This architecture supports:
- ✅ Modularity
- ✅ Extensibility  
- ✅ Scalability
- ✅ Maintainability
- ✅ Testability
