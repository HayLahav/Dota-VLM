"""
Vision-Language Model Annotation Generator
Supports: LLaVA, BLIP-2, Florence-2, and other VLMs
"""
import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
from PIL import Image


class VLMAnnotator:
    """Base class for Vision-Language Model annotation"""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        
    def load_model(self):
        raise NotImplementedError
        
    def generate_caption(self, image_path: str, prompt: str) -> str:
        raise NotImplementedError


class LLaVAAnnotator(VLMAnnotator):
    """LLaVA Vision-Language Model"""
    
    def __init__(self, model_name: str = 'llava-hf/llava-1.5-7b-hf', device: str = 'cuda'):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            print(f"Loading LLaVA model: {self.model_name}")
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading LLaVA: {e}")
            print("Trying alternative loading method...")
            self._load_llava_alternative()
    
    def _load_llava_alternative(self):
        """Alternative loading using llava library"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            self.model.eval()
            print(f"✓ Model loaded (alternative method)")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaVA model: {e}")
    
    def generate_caption(self, image_path: str, prompt: str, max_tokens: int = 256) -> str:
        """Generate text description for an image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""


class BLIP2Annotator(VLMAnnotator):
    """BLIP-2 Vision-Language Model"""
    
    def __init__(self, model_name: str = 'Salesforce/blip2-opt-2.7b', device: str = 'cuda'):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        print(f"Loading BLIP-2 model: {self.model_name}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def generate_caption(self, image_path: str, prompt: str, max_tokens: int = 256) -> str:
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""


class PromptTemplates:
    """Predefined prompt templates for different annotation tasks"""
    
    @staticmethod
    def object_attributes() -> str:
        return """Describe this object in detail. Include:
1. Size (small/medium/large)
2. Shape and orientation
3. Appearance and color
4. Activity or state (e.g., parked, moving, docked)
5. Any distinctive features
Be concise and factual."""
    
    @staticmethod
    def scene_caption() -> str:
        return """Describe this aerial image scene. Include:
1. Main objects and their arrangement
2. Overall context (urban, rural, water, etc.)
3. Notable patterns or structures
Keep it brief but informative."""
    
    @staticmethod
    def spatial_relations() -> str:
        return """Describe the spatial relationships in this image:
- Object positions (left, right, center, top, bottom)
- Arrangements (aligned, clustered, scattered)
- Proximity (next to, near, far from)
- Patterns (parallel, perpendicular, circular)"""
    
    @staticmethod
    def uncertainty_check() -> str:
        return """Assess this image quality and clarity:
1. Is the object clearly visible? (yes/no)
2. Any occlusions or ambiguities?
3. Confidence in identification (high/medium/low)
4. Any unusual or unclear aspects?"""
    
    @staticmethod
    def class_verification(class_name: str) -> str:
        return f"""Is this object a {class_name}? 
Answer with: YES or NO, followed by a brief explanation.
Consider the shape, context, and typical characteristics of a {class_name}."""


def generate_annotations_for_crop(
    vlm: VLMAnnotator,
    crop_path: str,
    crop_metadata: Dict,
    annotation_types: List[str]
) -> Dict:
    """Generate multiple types of annotations for a single crop"""
    
    annotations = {
        'crop_path': crop_path,
        'class_name': crop_metadata['class_name'],
        'detection_score': crop_metadata['score']
    }
    
    if 'attributes' in annotation_types:
        prompt = PromptTemplates.object_attributes()
        annotations['attributes'] = vlm.generate_caption(crop_path, prompt)
    
    if 'verification' in annotation_types:
        class_name = crop_metadata['class_name']
        prompt = PromptTemplates.class_verification(class_name)
        annotations['class_verification'] = vlm.generate_caption(crop_path, prompt, max_tokens=100)
    
    if 'uncertainty' in annotation_types:
        prompt = PromptTemplates.uncertainty_check()
        annotations['uncertainty'] = vlm.generate_caption(crop_path, prompt, max_tokens=150)
    
    return annotations


def process_crops(
    crops_dir: str,
    model_name: str,
    output_path: str,
    annotation_types: List[str] = ['attributes', 'verification'],
    model_type: str = 'llava'
):
    """Process all crops and generate VLM annotations"""
    
    # Load crops metadata
    crops_json = Path(crops_dir) / 'crops_metadata.json'
    if not crops_json.exists():
        raise FileNotFoundError(f"Crops metadata not found: {crops_json}")
    
    with open(crops_json, 'r') as f:
        crops_data = json.load(f)
    
    # Initialize VLM
    if model_type == 'llava':
        vlm = LLaVAAnnotator(model_name)
    elif model_type == 'blip2':
        vlm = BLIP2Annotator(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Process all crops
    all_annotations = {}
    total_crops = sum(img['num_crops'] for img in crops_data.values())
    
    print(f"\nGenerating annotations for {total_crops} objects...")
    
    pbar = tqdm(total=total_crops, desc="Annotating")
    
    for image_id, image_data in crops_data.items():
        image_annotations = []
        
        for crop_info in image_data['crops']:
            crop_path = Path(crops_dir) / crop_info['crop_path']
            
            if not crop_path.exists():
                pbar.update(1)
                continue
            
            try:
                annotations = generate_annotations_for_crop(
                    vlm=vlm,
                    crop_path=str(crop_path),
                    crop_metadata=crop_info,
                    annotation_types=annotation_types
                )
                
                # Merge with original metadata
                annotations.update({
                    'object_id': crop_info['object_id'],
                    'bbox': crop_info['bbox'],
                    'crop_size': crop_info['crop_size']
                })
                
                image_annotations.append(annotations)
                
            except Exception as e:
                print(f"\nError annotating {crop_path}: {e}")
            
            pbar.update(1)
        
        all_annotations[image_id] = {
            'image_path': image_data['image_path'],
            'annotations': image_annotations
        }
    
    pbar.close()
    
    # Save annotations
    with open(output_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\n✓ Annotations saved to {output_path}")
    print(f"  Images: {len(all_annotations)}")
    print(f"  Total annotations: {sum(len(d['annotations']) for d in all_annotations.values())}")


def main():
    parser = argparse.ArgumentParser(description='Generate VLM annotations for cropped objects')
    parser.add_argument('--crops_dir', type=str, required=True,
                        help='Directory containing crops and metadata')
    parser.add_argument('--model', type=str, default='llava',
                        choices=['llava', 'blip2'],
                        help='VLM model to use')
    parser.add_argument('--model_name', type=str, default='llava-hf/llava-1.5-7b-hf',
                        help='HuggingFace model name')
    parser.add_argument('--output', type=str, default='metadata.json',
                        help='Output JSON file')
    parser.add_argument('--annotations', nargs='+',
                        default=['attributes', 'verification'],
                        choices=['attributes', 'verification', 'uncertainty'],
                        help='Types of annotations to generate')
    
    args = parser.parse_args()
    
    process_crops(
        crops_dir=args.crops_dir,
        model_name=args.model_name,
        output_path=args.output,
        annotation_types=args.annotations,
        model_type=args.model
    )


if __name__ == '__main__':
    main()
