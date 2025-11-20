#!/usr/bin/env python3
"""
DOTA-VLM Main Pipeline
Orchestrates the complete workflow: detection -> cropping -> VLM annotation -> merging
"""
import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class DOTAVLMPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.output_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    
    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_step(self, step_name: str, command: list) -> bool:
        """Run a pipeline step and handle errors"""
        self.log(f"\n{'='*60}")
        self.log(f"STEP: {step_name}")
        self.log(f"{'='*60}")
        self.log(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            
            self.log(result.stdout)
            if result.stderr:
                self.log(f"Warnings: {result.stderr}")
            
            self.log(f"✓ {step_name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"✗ {step_name} failed!")
            self.log(f"Error: {e.stderr}")
            
            if not self.config['processing'].get('skip_on_error', False):
                self.log("Pipeline aborted.")
                sys.exit(1)
            
            return False
    
    def step_1_detection(self) -> bool:
        """Step 1: Run object detection"""
        det_config = self.config['detection']
        data_config = self.config['data']
        
        command = [
            'python', 'detection/run_detector.py',
            '--input_dir', data_config['dota_images_dir'],
            '--output', data_config['detections_json'],
            '--model_path', det_config['model_path'],
            '--detector', det_config['model_type'],
            '--conf_threshold', str(det_config['conf_threshold'])
        ]
        
        return self.run_step("Object Detection", command)
    
    def step_2_cropping(self) -> bool:
        """Step 2: Crop detected objects"""
        crop_config = self.config['cropping']
        data_config = self.config['data']
        
        command = [
            'python', 'tools/crop_objects.py',
            '--detections', data_config['detections_json'],
            '--images_dir', data_config['dota_images_dir'],
            '--out_dir', data_config['crops_dir'],
            '--padding', str(crop_config['padding'])
        ]
        
        return self.run_step("Object Cropping", command)
    
    def step_3_vlm_annotation(self) -> bool:
        """Step 3: Generate VLM annotations"""
        vlm_config = self.config['vlm']
        data_config = self.config['data']
        
        command = [
            'python', 'vlm/generate_annotations.py',
            '--crops_dir', data_config['crops_dir'],
            '--model', vlm_config['model_type'],
            '--model_name', vlm_config['model_name'],
            '--output', data_config['metadata_json'],
            '--annotations', *vlm_config['annotation_types']
        ]
        
        return self.run_step("VLM Annotation Generation", command)
    
    def step_4_merging(self) -> bool:
        """Step 4: Merge annotations"""
        merge_config = self.config['merging']
        data_config = self.config['data']
        
        command = [
            'python', 'tools/merge_annotations.py',
            '--dota_json', data_config['detections_json'],
            '--vlm_json', data_config['metadata_json'],
            '--output', data_config['final_json']
        ]
        
        if merge_config.get('include_unverified', True):
            command.append('--include_unverified')
        
        if merge_config.get('generate_summary', True):
            command.extend(['--summary', merge_config.get('summary_path', '')])
        
        return self.run_step("Annotation Merging", command)
    
    def validate_inputs(self) -> bool:
        """Validate that all required inputs exist"""
        self.log("Validating inputs...")
        
        # Check images directory
        images_dir = Path(self.config['data']['dota_images_dir'])
        if not images_dir.exists():
            self.log(f"✗ Images directory not found: {images_dir}")
            return False
        
        # Check model checkpoint
        model_path = Path(self.config['detection']['model_path'])
        if not model_path.exists():
            self.log(f"✗ Model checkpoint not found: {model_path}")
            self.log("  Please download or train a YOLO-OBB model first.")
            return False
        
        self.log("✓ All inputs validated")
        return True
    
    def run_pipeline(self, steps: list = None):
        """Run the complete pipeline or specific steps"""
        start_time = datetime.now()
        
        self.log("\n" + "="*60)
        self.log("DOTA-VLM PIPELINE START")
        self.log("="*60)
        self.log(f"Start time: {start_time}")
        self.log(f"Config: {self.config}")
        
        # Validate inputs
        if not self.validate_inputs():
            self.log("Pipeline aborted due to validation errors.")
            sys.exit(1)
        
        # Define all steps
        all_steps = {
            'detection': self.step_1_detection,
            'cropping': self.step_2_cropping,
            'vlm': self.step_3_vlm_annotation,
            'merging': self.step_4_merging
        }
        
        # Run requested steps
        if steps is None:
            steps = list(all_steps.keys())
        
        results = {}
        for step_name in steps:
            if step_name not in all_steps:
                self.log(f"✗ Unknown step: {step_name}")
                continue
            
            results[step_name] = all_steps[step_name]()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.log("\n" + "="*60)
        self.log("PIPELINE SUMMARY")
        self.log("="*60)
        
        for step_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            self.log(f"{step_name.upper()}: {status}")
        
        self.log(f"\nTotal duration: {duration}")
        self.log(f"Log saved to: {self.log_file}")
        
        if all(results.values()):
            self.log("\n🎉 Pipeline completed successfully!")
            self.log(f"Output: {self.config['data']['final_json']}")
        else:
            self.log("\n⚠️  Pipeline completed with errors.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='DOTA-VLM Pipeline - Automated aerial imagery annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py --config configs/config.yaml
  
  # Run specific steps
  python pipeline.py --config configs/config.yaml --steps detection cropping
  
  # Run from cropping onwards
  python pipeline.py --config configs/config.yaml --steps cropping vlm merging
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['detection', 'cropping', 'vlm', 'merging'],
        default=None,
        help='Specific steps to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a config file or use the default: configs/config.yaml")
        sys.exit(1)
    
    # Run pipeline
    pipeline = DOTAVLMPipeline(args.config)
    pipeline.run_pipeline(steps=args.steps)


if __name__ == '__main__':
    main()
