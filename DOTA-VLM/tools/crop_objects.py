"""
Crop objects from images using oriented bounding boxes
Handles rotation and padding for clean crops
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.affinity import rotate


def get_rotated_crop(
    image: np.ndarray,
    bbox: List[float],
    padding: int = 10
) -> np.ndarray:
    """
    Crop object from image using oriented bounding box
    
    Args:
        image: Input image (H, W, C)
        bbox: [cx, cy, w, h, angle] format
        padding: Padding around the crop in pixels
    
    Returns:
        Cropped and rotated image patch
    """
    cx, cy, w, h, angle = bbox
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Rotate the entire image
    img_h, img_w = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
    
    # Calculate crop boundaries with padding
    x1 = int(cx - w/2 - padding)
    y1 = int(cy - h/2 - padding)
    x2 = int(cx + w/2 + padding)
    y2 = int(cy + h/2 + padding)
    
    # Ensure boundaries are within image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    # Crop the rotated region
    crop = rotated[y1:y2, x1:x2]
    
    return crop


def get_rotated_crop_from_corners(
    image: np.ndarray,
    corners: List[List[float]],
    padding: int = 10
) -> np.ndarray:
    """
    Crop using 4 corner points directly
    
    Args:
        image: Input image
        corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        padding: Padding in pixels
    
    Returns:
        Cropped image patch
    """
    corners = np.array(corners, dtype=np.float32)
    
    # Find bounding rectangle
    rect = cv2.minAreaRect(corners)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get width and height
    width = int(rect[1][0]) + 2 * padding
    height = int(rect[1][1]) + 2 * padding
    
    # Destination points for perspective transform
    dst_pts = np.array([
        [padding, height - padding],
        [padding, padding],
        [width - padding, padding],
        [width - padding, height - padding]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    
    # Warp the image
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def crop_detections(
    image_path: str,
    detections: List[Dict],
    output_dir: str,
    image_id: str,
    padding: int = 10,
    use_corners: bool = True
) -> List[Dict]:
    """
    Crop all detected objects from an image
    
    Returns:
        List of crop metadata with paths
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Create output directory
    image_output_dir = Path(output_dir) / image_id
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    crops_metadata = []
    
    for idx, det in enumerate(detections):
        try:
            # Crop object
            if use_corners and 'bbox_corners' in det:
                crop = get_rotated_crop_from_corners(
                    image, det['bbox_corners'], padding
                )
            else:
                crop = get_rotated_crop(
                    image, det['bbox'], padding
                )
            
            # Skip very small crops
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # Save crop
            crop_filename = f"obj_{idx:04d}_{det['class_name']}.jpg"
            crop_path = image_output_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            # Store metadata
            crops_metadata.append({
                'crop_path': str(crop_path.relative_to(output_dir)),
                'object_id': idx,
                'bbox': det['bbox'],
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'score': det['score'],
                'crop_size': [crop.shape[1], crop.shape[0]]  # width, height
            })
            
        except Exception as e:
            print(f"  Error cropping object {idx}: {e}")
            continue
    
    return crops_metadata


def process_all_detections(
    detections_json: str,
    images_dir: str,
    output_dir: str,
    padding: int = 10
):
    """Process all detections and create crops"""
    
    # Load detections
    with open(detections_json, 'r') as f:
        all_detections = json.load(f)
    
    print(f"Processing {len(all_detections)} images...")
    
    # Store all crop metadata
    crops_database = {}
    
    for image_id, image_data in tqdm(all_detections.items(), desc="Cropping objects"):
        image_path = Path(images_dir) / image_data['image_path']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            crops = crop_detections(
                image_path=str(image_path),
                detections=image_data['detections'],
                output_dir=output_dir,
                image_id=image_id.replace('/', '_').replace('\\', '_'),
                padding=padding
            )
            
            crops_database[image_id] = {
                'image_path': image_data['image_path'],
                'num_crops': len(crops),
                'crops': crops
            }
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue
    
    # Save crops database
    output_json = Path(output_dir) / 'crops_metadata.json'
    with open(output_json, 'w') as f:
        json.dump(crops_database, f, indent=2)
    
    print(f"\n✓ Cropping complete!")
    print(f"  Images processed: {len(crops_database)}")
    print(f"  Total crops: {sum(d['num_crops'] for d in crops_database.values())}")
    print(f"  Metadata saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Crop objects from DOTA images')
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing original images')
    parser.add_argument('--out_dir', type=str, default='crops',
                        help='Output directory for crops')
    parser.add_argument('--padding', type=int, default=10,
                        help='Padding around crops in pixels')
    
    args = parser.parse_args()
    
    process_all_detections(
        detections_json=args.detections,
        images_dir=args.images_dir,
        output_dir=args.out_dir,
        padding=args.padding
    )


if __name__ == '__main__':
    main()
