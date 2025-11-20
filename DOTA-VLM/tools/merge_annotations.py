"""
Merge detection annotations with VLM-generated metadata
Export to COCO-style JSON format
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class COCOAnnotationBuilder:
    """Build COCO-style annotations with VLM metadata"""
    
    def __init__(self):
        self.categories = []
        self.images = []
        self.annotations = []
        self.category_id_map = {}
        
    def add_categories(self, class_names: List[str]):
        """Add DOTA categories"""
        for idx, name in enumerate(sorted(set(class_names))):
            self.categories.append({
                'id': idx,
                'name': name,
                'supercategory': 'object'
            })
            self.category_id_map[name] = idx
    
    def add_image(self, image_id: int, filename: str, width: int, height: int) -> int:
        """Add image metadata"""
        self.images.append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height,
            'date_captured': datetime.now().isoformat()
        })
        return image_id
    
    def add_annotation(
        self,
        ann_id: int,
        image_id: int,
        category_name: str,
        bbox: List[float],
        detection_score: float,
        vlm_metadata: Dict = None
    ):
        """Add annotation with optional VLM metadata"""
        
        # Convert oriented bbox to COCO format
        cx, cy, w, h, angle = bbox
        x = cx - w/2
        y = cy - h/2
        
        annotation = {
            'id': ann_id,
            'image_id': image_id,
            'category_id': self.category_id_map[category_name],
            'bbox': [x, y, w, h],  # COCO format: [x, y, width, height]
            'area': w * h,
            'iscrowd': 0,
            'segmentation': [],  # Can add polygon if needed
            
            # Oriented bbox information
            'oriented_bbox': {
                'center': [cx, cy],
                'size': [w, h],
                'angle': angle
            },
            
            # Detection metadata
            'detection_score': detection_score,
        }
        
        # Add VLM metadata if available
        if vlm_metadata:
            annotation['vlm_metadata'] = {
                'attributes': vlm_metadata.get('attributes', ''),
                'class_verification': vlm_metadata.get('class_verification', ''),
                'uncertainty': vlm_metadata.get('uncertainty', ''),
                'crop_size': vlm_metadata.get('crop_size', [0, 0])
            }
        
        self.annotations.append(annotation)
    
    def build(self) -> Dict:
        """Build final COCO-style JSON"""
        return {
            'info': {
                'description': 'DOTA-VLM Dataset',
                'version': '1.0',
                'year': 2024,
                'contributor': 'DOTA-VLM Pipeline',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'categories': self.categories,
            'images': self.images,
            'annotations': self.annotations
        }


def merge_annotations(
    detections_json: str,
    vlm_json: str,
    output_path: str,
    include_unverified: bool = True
):
    """
    Merge detection results with VLM annotations
    
    Args:
        detections_json: Path to detection results
        vlm_json: Path to VLM annotations
        output_path: Output path for merged JSON
        include_unverified: Include objects without VLM annotations
    """
    
    # Load detections
    with open(detections_json, 'r') as f:
        detections = json.load(f)
    
    # Load VLM annotations
    with open(vlm_json, 'r') as f:
        vlm_annotations = json.load(f)
    
    # Initialize COCO builder
    builder = COCOAnnotationBuilder()
    
    # Collect all category names
    all_categories = set()
    for img_data in detections.values():
        for det in img_data['detections']:
            all_categories.add(det['class_name'])
    
    builder.add_categories(list(all_categories))
    
    # Process each image
    image_id = 0
    annotation_id = 0
    
    print(f"Merging {len(detections)} images...")
    
    for image_key, det_data in detections.items():
        # Add image
        img_size = det_data['image_size']
        image_id = builder.add_image(
            image_id=image_id,
            filename=det_data['image_path'],
            width=img_size[0],
            height=img_size[1]
        )
        
        # Get VLM annotations for this image
        vlm_data = vlm_annotations.get(image_key, {})
        vlm_anns_list = vlm_data.get('annotations', [])
        
        # Create lookup by object_id
        vlm_by_obj_id = {
            ann['object_id']: ann
            for ann in vlm_anns_list
        }
        
        # Add annotations
        for idx, detection in enumerate(det_data['detections']):
            vlm_metadata = vlm_by_obj_id.get(idx)
            
            # Skip if no VLM metadata and we're not including unverified
            if not include_unverified and not vlm_metadata:
                continue
            
            builder.add_annotation(
                ann_id=annotation_id,
                image_id=image_id,
                category_name=detection['class_name'],
                bbox=detection['bbox'],
                detection_score=detection['score'],
                vlm_metadata=vlm_metadata
            )
            
            annotation_id += 1
        
        image_id += 1
    
    # Build and save
    coco_data = builder.build()
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ DOTA-VLM annotations saved!")
    print(f"  Output: {output_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Calculate statistics
    vlm_enhanced = sum(
        1 for ann in coco_data['annotations']
        if 'vlm_metadata' in ann
    )
    print(f"  VLM-enhanced: {vlm_enhanced} ({vlm_enhanced/len(coco_data['annotations'])*100:.1f}%)")


def create_summary_report(
    dota_vlm_json: str,
    output_txt: str = None
):
    """Generate a summary report of the DOTA-VLM dataset"""
    
    with open(dota_vlm_json, 'r') as f:
        data = json.load(f)
    
    report = []
    report.append("=" * 60)
    report.append("DOTA-VLM Dataset Summary")
    report.append("=" * 60)
    report.append("")
    
    # Basic stats
    report.append(f"Total Images: {len(data['images'])}")
    report.append(f"Total Annotations: {len(data['annotations'])}")
    report.append(f"Categories: {len(data['categories'])}")
    report.append("")
    
    # Category distribution
    report.append("Category Distribution:")
    cat_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        report.append(f"  {cat_name}: {count}")
    report.append("")
    
    # VLM metadata stats
    vlm_count = sum(1 for ann in data['annotations'] if 'vlm_metadata' in ann)
    report.append(f"VLM-Enhanced Annotations: {vlm_count} ({vlm_count/len(data['annotations'])*100:.1f}%)")
    report.append("")
    
    # Sample annotations
    report.append("Sample VLM Annotations:")
    report.append("-" * 60)
    for ann in data['annotations'][:3]:
        if 'vlm_metadata' in ann:
            cat_name = next(c['name'] for c in data['categories'] if c['id'] == ann['category_id'])
            report.append(f"\nCategory: {cat_name}")
            report.append(f"Detection Score: {ann['detection_score']:.3f}")
            report.append(f"Attributes: {ann['vlm_metadata']['attributes'][:150]}...")
            report.append("")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_txt:
        with open(output_txt, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Summary report saved to {output_txt}")


def main():
    parser = argparse.ArgumentParser(description='Merge DOTA detections with VLM annotations')
    parser.add_argument('--dota_json', type=str, required=True,
                        help='Path to detections JSON')
    parser.add_argument('--vlm_json', type=str, required=True,
                        help='Path to VLM annotations JSON')
    parser.add_argument('--output', type=str, default='dota_vlm.json',
                        help='Output COCO-style JSON path')
    parser.add_argument('--include_unverified', action='store_true',
                        help='Include objects without VLM annotations')
    parser.add_argument('--summary', type=str, default=None,
                        help='Generate summary report (optional output path)')
    
    args = parser.parse_args()
    
    # Merge annotations
    merge_annotations(
        detections_json=args.dota_json,
        vlm_json=args.vlm_json,
        output_path=args.output,
        include_unverified=args.include_unverified
    )
    
    # Generate summary if requested
    if args.summary or args.summary == '':
        summary_path = args.summary if args.summary else 'dota_vlm_summary.txt'
        create_summary_report(args.output, summary_path)


if __name__ == '__main__':
    main()
