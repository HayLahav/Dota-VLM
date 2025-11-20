# DOTA-VLM Quick Start Guide

##  Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DOTA-VLM.git
cd DOTA-VLM

# Create environment
conda create -n dota_vlm python=3.10 -y
conda activate dota_vlm

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Create data directory
mkdir -p data/DOTA/images

# Download DOTA dataset
# Visit: https://captain-whu.github.io/DOTA/dataset.html
# Extract images to data/DOTA/images/
```

### 3. Get Pre-trained Model

**Option A: Use Pre-trained YOLO-OBB**
```bash
# Download from Ultralytics
mkdir -p checkpoints
# Place your trained YOLO-OBB model in checkpoints/
```

**Option B: Train Your Own**
```bash
# Follow YOLO-OBB training guide
# https://docs.ultralytics.com/tasks/obb/
```

### 4. Run Pipeline

```bash
# Run complete pipeline with one command
python pipeline.py --config configs/config.yaml
```

That's it! Your annotations will be in `outputs/dota_vlm.json`

##  Step-by-Step Guide

### Step 1: Detection (5-10 min)

```bash
python detection/run_detector.py \
    --input_dir data/DOTA/images \
    --output detections.json \
    --model_path checkpoints/yolo_obb.pt \
    --conf_threshold 0.3
```

**Output:** `detections.json` with oriented bounding boxes

### Step 2: Cropping (2-5 min)

```bash
python tools/crop_objects.py \
    --detections detections.json \
    --images_dir data/DOTA/images \
    --out_dir crops/
```

**Output:** `crops/` directory with individual object patches

### Step 3: VLM Annotation (15-30 min, depending on GPU)

```bash
python vlm/generate_annotations.py \
    --crops_dir crops/ \
    --model llava \
    --output metadata.json \
    --annotations attributes verification
```

**Output:** `metadata.json` with VLM-generated descriptions

### Step 4: Merge (< 1 min)

```bash
python tools/merge_annotations.py \
    --dota_json detections.json \
    --vlm_json metadata.json \
    --output dota_vlm.json \
    --summary
```

**Output:** `dota_vlm.json` in COCO format

##  Common Use Cases

### Use Case 1: Quick Test on Sample Images

```bash
# Test on just a few images
python detection/run_detector.py \
    --input_dir data/DOTA/samples \
    --output test_detections.json \
    --model_path checkpoints/yolo_obb.pt

# Process only those
python tools/crop_objects.py \
    --detections test_detections.json \
    --images_dir data/DOTA/samples \
    --out_dir test_crops/
```

### Use Case 2: High-Quality Annotations Only

```bash
# Use higher confidence threshold
python detection/run_detector.py \
    --input_dir data/DOTA/images \
    --conf_threshold 0.5 \
    --output high_conf_detections.json \
    --model_path checkpoints/yolo_obb.pt
```

### Use Case 3: Specific Categories Only

Edit `configs/config.yaml`:

```yaml
filtering:
  include_categories: ['airplane', 'ship', 'vehicle']
```

### Use Case 4: Use Different VLM Models

```bash
# BLIP-2
python vlm/generate_annotations.py \
    --crops_dir crops/ \
    --model blip2 \
    --model_name Salesforce/blip2-opt-2.7b \
    --output metadata_blip2.json

# Larger LLaVA
python vlm/generate_annotations.py \
    --crops_dir crops/ \
    --model llava \
    --model_name llava-hf/llava-1.5-13b-hf \
    --output metadata_llava13b.json
```

## 🔍 Viewing Results

### Visualize Detections

```bash
python tools/visualize.py detections \
    --image data/DOTA/images/P0001.png \
    --detections detections.json \
    --output viz_detections.jpg
```

### Visualize VLM Annotations

```bash
python tools/visualize.py vlm \
    --image data/DOTA/images/P0001.png \
    --dota_vlm_json dota_vlm.json \
    --image_id 0 \
    --output viz_vlm.jpg
```

### Create Preview Grid

```bash
python tools/visualize.py grid \
    --dota_vlm_json dota_vlm.json \
    --images_dir data/DOTA/images \
    --output_dir previews/ \
    --num_samples 20
```

##  Troubleshooting

### Problem: CUDA Out of Memory

**Solution:**
```bash
# Use smaller model
python vlm/generate_annotations.py \
    --model llava \
    --model_name llava-hf/llava-1.5-7b-hf  # Instead of 13b
```

### Problem: Slow VLM Inference

**Solutions:**
1. Use GPU (check with `nvidia-smi`)
2. Use quantization (edit config: `use_8bit: true`)
3. Process in batches

### Problem: Model Not Found

**Solution:**
```bash
# Model will download automatically on first run
# Or manually download:
from transformers import LlavaNextProcessor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
```

### Problem: Import Errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install ultralytics transformers --upgrade
```

##  Tips & Best Practices

### Performance Tips

1. **Use GPU** - VLM annotation is 10-100x faster on GPU
2. **Batch processing** - Process multiple images at once
3. **Start small** - Test on 10-20 images first
4. **Cache crops** - Reuse crops for different VLM models

### Quality Tips

1. **Tune confidence threshold** - Start at 0.3, adjust based on results
2. **Use verification** - VLM can validate detections
3. **Check samples** - Always visualize a few results
4. **Iterate prompts** - Customize prompts for your domain

### Workflow Tips

1. **Save intermediate results** - Enable in config
2. **Version control** - Track configuration changes
3. **Document experiments** - Keep notes on what works
4. **Use jupyter notebook** - See `examples/tutorial.ipynb`

##  Expected Performance

On a typical setup (NVIDIA RTX 3090, 24GB VRAM):

| Step | Time (100 images) | GPU Memory |
|------|------------------|------------|
| Detection | 2-5 min | 2-4 GB |
| Cropping | 1-2 min | N/A |
| VLM (LLaVA-7B) | 15-30 min | 8-12 GB |
| Merging | < 1 min | N/A |

**Total:** ~20-40 minutes for 100 images

##  Next Steps

1. **Try the tutorial notebook:** `examples/tutorial.ipynb`
2. **Customize prompts:** Edit `vlm/generate_annotations.py`
3. **Train detector:** Improve detection on your data
4. **Scale up:** Process full DOTA dataset
5. **Downstream tasks:** Use enriched annotations for training

##  Resources

- [DOTA Dataset](https://captain-whu.github.io/DOTA/dataset.html)
- [YOLO-OBB Docs](https://docs.ultralytics.com/tasks/obb/)
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [Transformers Docs](https://huggingface.co/docs/transformers/)

## Getting Help

- Open an issue on GitHub
- Check existing issues
- Read the full README.md
- Try the tutorial notebook


