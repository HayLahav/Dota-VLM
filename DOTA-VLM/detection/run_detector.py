"""
Oriented Object Detector for DOTA Dataset
Supports YOLO-OBB, Oriented RCNN, and other rotated box detectors
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm


class OrientedObjectDetector:
    """Base class for oriented object detection"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
    def load_model(self):
        raise NotImplementedError
        
    def detect(self, image_path: str) -> List[Dict]:
        raise NotImplementedError


class YOLOOBBDetector(OrientedObjectDetector):
    """YOLO Oriented Bounding Box Detector"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        super().__init__(model_path, conf_threshold)
        self.load_model()
        
    def load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"✓ Loaded YOLO-OBB model from {self.model_path}")
        except ImportError:
            print("ERROR: ultralytics not installed. Run: pip install ultralytics")
            raise
            
    def detect(self, image_path: str) -> List[Dict]:
        """
        Run detection and return oriented bounding boxes
        Returns: List of detections with format:
        {
            'bbox': [x, y, w, h, angle],  # Oriented bbox
            'score': float,
            'class_id': int,
            'class_name': str
        }
        """
        results = self.model(image_path, conf=self.conf_threshold)
        detections = []
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # Oriented bounding boxes
                boxes = result.obb.xyxyxyxy.cpu().numpy()  # 4 corner points
                scores = result.obb.conf.cpu().numpy()
                class_ids = result.obb.cls.cpu().numpy().astype(int)
                
                for box, score, class_id in zip(boxes, scores, class_ids):
                    # Convert 4-point format to [cx, cy, w, h, angle]
                    obb = self._convert_to_obb_format(box)
                    
                    detections.append({
                        'bbox': obb,
                        'bbox_corners': box.tolist(),  # Keep original corners
                        'score': float(score),
                        'class_id': int(class_id),
                        'class_name': result.names[class_id]
                    })
            elif hasattr(result, 'boxes'):
                # Fallback to regular bounding boxes
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    
                    detections.append({
                        'bbox': [cx, cy, w, h, 0.0],  # angle = 0 for axis-aligned
                        'score': float(score),
                        'class_id': int(class_id),
                        'class_name': result.names[class_id]
                    })
        
        return detections
    
    def _convert_to_obb_format(self, corners: np.ndarray) -> List[float]:
        """Convert 4 corner points to [cx, cy, w, h, angle] format"""
        # corners shape: (4, 2)
        cx = corners[:, 0].mean()
        cy = corners[:, 1].mean()
        
        # Compute width and height from corner distances
        edge1 = np.linalg.norm(corners[1] - corners[0])
        edge2 = np.linalg.norm(corners[2] - corners[1])
        w = max(edge1, edge2)
        h = min(edge1, edge2)
        
        # Compute angle from first edge
        dx = corners[1, 0] - corners[0, 0]
        dy = corners[1, 1] - corners[0, 1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        return [float(cx), float(cy), float(w), float(h), float(angle)]


def process_directory(
    input_dir: str,
    output_path: str,
    model_path: str,
    detector_type: str = 'yolo_obb',
    conf_threshold: float = 0.3
):
    """Process all images in a directory and save detections"""
    
    # Initialize detector
    if detector_type == 'yolo_obb':
        detector = YOLOOBBDetector(model_path, conf_threshold)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    # Get all images
    image_dir = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [
        p for p in image_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    all_detections = {}
    
    for img_path in tqdm(image_paths, desc="Detecting objects"):
        try:
            detections = detector.detect(str(img_path))
            
            # Store results
            rel_path = str(img_path.relative_to(image_dir))
            all_detections[rel_path] = {
                'image_path': rel_path,
                'image_size': get_image_size(str(img_path)),
                'detections': detections,
                'num_objects': len(detections)
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"✓ Saved detections to {output_path}")
    print(f"  Total images: {len(all_detections)}")
    print(f"  Total objects: {sum(d['num_objects'] for d in all_detections.values())}")


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions (width, height)"""
    img = cv2.imread(image_path)
    if img is None:
        return (0, 0)
    return (img.shape[1], img.shape[0])


def main():
    parser = argparse.ArgumentParser(description='Run oriented object detection on DOTA images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing DOTA images')
    parser.add_argument('--output', type=str, default='detections.json',
                        help='Output JSON file path')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to detection model checkpoint')
    parser.add_argument('--detector', type=str, default='yolo_obb',
                        choices=['yolo_obb', 'oriented_rcnn'],
                        help='Detector type')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    process_directory(
        input_dir=args.input_dir,
        output_path=args.output,
        model_path=args.model_path,
        detector_type=args.detector,
        conf_threshold=args.conf_threshold
    )


if __name__ == '__main__':
    main()
