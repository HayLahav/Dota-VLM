"""
Visualization utilities for DOTA-VLM
Draw bounding boxes, display annotations, create preview images
"""
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPLPolygon


def rotate_box(cx, cy, w, h, angle):
    """Convert center + angle bbox to 4 corner points"""
    angle_rad = angle * np.pi / 180
    
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate and translate
    rotated = corners @ rot_matrix.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    return rotated.astype(np.int32)


def draw_oriented_box(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw oriented bounding box on image"""
    cx, cy, w, h, angle = bbox
    
    # Get corner points
    corners = rotate_box(cx, cy, w, h, angle)
    
    # Draw box
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=thickness)
    
    # Draw center point
    cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
    
    return image


def draw_detection_label(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    score: float,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw class label and confidence score"""
    cx, cy, w, h, angle = bbox
    
    # Position label above box
    text = f"{label}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Background rectangle
    x1 = int(cx - text_w // 2)
    y1 = int(cy - h // 2 - text_h - 5)
    x2 = x1 + text_w
    y2 = y1 + text_h + 5
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        image, text,
        (x1, y1 + text_h),
        font, font_scale, (255, 255, 255), thickness
    )
    
    return image


def visualize_detections(
    image_path: str,
    detections: List[Dict],
    output_path: str = None,
    draw_labels: bool = True,
    color_by_class: bool = True
) -> np.ndarray:
    """Visualize all detections on an image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Define colors for different classes
    np.random.seed(42)
    class_colors = {}
    
    for det in detections:
        class_name = det['class_name']
        
        if color_by_class:
            if class_name not in class_colors:
                class_colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
            color = class_colors[class_name]
        else:
            # Color by confidence
            score = det['score']
            if score > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif score > 0.4:
                color = (255, 165, 0)  # Orange - medium confidence
            else:
                color = (255, 0, 0)  # Red - low confidence
        
        # Draw bounding box
        image = draw_oriented_box(image, det['bbox'], color, thickness=2)
        
        # Draw label
        if draw_labels:
            image = draw_detection_label(
                image, det['bbox'],
                det['class_name'],
                det['score'],
                color
            )
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"✓ Saved visualization to {output_path}")
    
    return image


def visualize_vlm_annotations(
    image_path: str,
    annotations: List[Dict],
    output_path: str = None,
    max_text_length: int = 100
):
    """Create visualization with VLM annotations"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_rgb)
    
    # Draw each annotation
    for idx, ann in enumerate(annotations):
        bbox = ann['bbox']
        cx, cy, w, h, angle = bbox
        
        # Get color based on score
        score = ann.get('detection_score', 0)
        if score > 0.7:
            color = 'green'
        elif score > 0.4:
            color = 'orange'
        else:
            color = 'red'
        
        # Draw rotated box
        corners = rotate_box(cx, cy, w, h, angle)
        poly = MPLPolygon(corners, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(poly)
        
        # Add annotation text
        if 'vlm_metadata' in ann and ann['vlm_metadata'].get('attributes'):
            text = ann['vlm_metadata']['attributes']
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            ax.text(
                cx, cy - h/2 - 10,
                f"{ann['class_name']}\n{text}",
                color='white',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                verticalalignment='bottom',
                horizontalalignment='center'
            )
        else:
            ax.text(
                cx, cy - h/2 - 10,
                f"{ann['class_name']} ({score:.2f})",
                color='white',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                verticalalignment='bottom',
                horizontalalignment='center'
            )
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved VLM visualization to {output_path}")
        plt.close()
    else:
        plt.show()


def create_preview_grid(
    dota_vlm_json: str,
    images_dir: str,
    output_dir: str,
    num_samples: int = 10
):
    """Create a grid of preview images with annotations"""
    # Load annotations
    with open(dota_vlm_json, 'r') as f:
        data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    import random
    images = random.sample(data['images'], min(num_samples, len(data['images'])))
    
    print(f"Creating previews for {len(images)} images...")
    
    for img_info in images:
        image_path = Path(images_dir) / img_info['file_name']
        
        if not image_path.exists():
            continue
        
        # Get annotations for this image
        img_annotations = [
            ann for ann in data['annotations']
            if ann['image_id'] == img_info['id']
        ]
        
        # Create visualization
        output_path = output_dir / f"preview_{img_info['id']:04d}.jpg"
        visualize_vlm_annotations(
            str(image_path),
            img_annotations,
            str(output_path),
            max_text_length=80
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize DOTA-VLM annotations')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization commands')
    
    # Detections visualization
    det_parser = subparsers.add_parser('detections', help='Visualize detections only')
    det_parser.add_argument('--image', type=str, required=True)
    det_parser.add_argument('--detections', type=str, required=True)
    det_parser.add_argument('--output', type=str, default='detection_viz.jpg')
    
    # VLM annotations visualization
    vlm_parser = subparsers.add_parser('vlm', help='Visualize VLM annotations')
    vlm_parser.add_argument('--image', type=str, required=True)
    vlm_parser.add_argument('--dota_vlm_json', type=str, required=True)
    vlm_parser.add_argument('--image_id', type=int, required=True)
    vlm_parser.add_argument('--output', type=str, default='vlm_viz.jpg')
    
    # Preview grid
    grid_parser = subparsers.add_parser('grid', help='Create preview grid')
    grid_parser.add_argument('--dota_vlm_json', type=str, required=True)
    grid_parser.add_argument('--images_dir', type=str, required=True)
    grid_parser.add_argument('--output_dir', type=str, default='previews')
    grid_parser.add_argument('--num_samples', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == 'detections':
        with open(args.detections, 'r') as f:
            detections_data = json.load(f)
        
        # Find detections for the image
        for img_key, img_data in detections_data.items():
            if img_data['image_path'] in args.image or args.image.endswith(img_data['image_path']):
                visualize_detections(
                    args.image,
                    img_data['detections'],
                    args.output
                )
                break
    
    elif args.command == 'vlm':
        with open(args.dota_vlm_json, 'r') as f:
            data = json.load(f)
        
        # Get annotations for image_id
        annotations = [
            ann for ann in data['annotations']
            if ann['image_id'] == args.image_id
        ]
        
        visualize_vlm_annotations(args.image, annotations, args.output)
    
    elif args.command == 'grid':
        create_preview_grid(
            args.dota_vlm_json,
            args.images_dir,
            args.output_dir,
            args.num_samples
        )


if __name__ == '__main__':
    main()
