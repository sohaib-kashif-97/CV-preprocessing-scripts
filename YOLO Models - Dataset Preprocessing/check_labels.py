#!/usr/bin/env python3
"""
Dataset Validation Script for YOLO Training
Checks for:
- Missing label files for images
- Empty label files
- Invalid YOLO format annotations
- Invalid class IDs
- Image/label correspondence
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import cv2


class DatasetValidator:
    def __init__(self, dataset_path, num_classes=6):
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.issues = defaultdict(list)
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'images_without_labels': 0,
            'empty_label_files': 0,
            'invalid_format': 0,
            'valid_annotations': 0,
            'total_objects': 0
        }
    
    def validate_split(self, split_name):
        """Validate a single split (train/val/test)"""
        print(f"\n{'='*70}")
        print(f"Validating {split_name.upper()} Split")
        print('='*70)
        
        images_dir = self.dataset_path / split_name / 'images'
        labels_dir = self.dataset_path / split_name / 'labels'
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        if not labels_dir.exists():
            print(f"❌ Labels directory not found: {labels_dir}")
            return False
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = {
            f.stem: f for f in images_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        }
        
        label_files = {
            f.stem: f for f in labels_dir.iterdir() 
            if f.suffix == '.txt'
        }
        
        print(f"\n📊 Found {len(image_files)} images and {len(label_files)} label files")
        
        split_stats = {
            'split_name': split_name,
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'images_without_labels': 0,
            'empty_label_files': 0,
            'invalid_format': 0,
            'valid_images': 0,
            'total_objects': 0,
            'objects_per_image': []
        }
        
        # Check for images without labels
        print(f"\n🔍 Checking image-label correspondence...")
        missing_labels = []
        for img_stem in image_files:
            if img_stem not in label_files:
                missing_labels.append(img_stem)
                split_stats['images_without_labels'] += 1
        
        if missing_labels:
            print(f"⚠️  Found {len(missing_labels)} images WITHOUT label files:")
            for stem in missing_labels[:10]:  # Show first 10
                print(f"   - {stem}")
            if len(missing_labels) > 10:
                print(f"   ... and {len(missing_labels) - 10} more")
            self.issues[split_name].extend(missing_labels)
        else:
            print(f"✅ All {len(image_files)} images have corresponding label files")
        
        # Validate each label file
        print(f"\n🔍 Validating label file format...")
        for img_stem, img_path in image_files.items():
            if img_stem not in label_files:
                continue
            
            label_path = label_files[img_stem]
            
            # Check if label file is empty
            if label_path.stat().st_size == 0:
                print(f"⚠️  Empty label file: {img_stem}.txt")
                split_stats['empty_label_files'] += 1
                self.issues[f"{split_name}_empty"].append(img_stem)
                continue
            
            # Validate format
            try:
                objects_in_image = 0
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        
                        # Check format: class_id x_center y_center width height
                        if len(parts) != 5:
                            print(f"❌ Invalid format in {img_stem}.txt (line {line_num})")
                            print(f"   Expected 5 values, got {len(parts)}: {line}")
                            split_stats['invalid_format'] += 1
                            break
                        
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Validate class ID
                            if class_id < 0 or class_id >= self.num_classes:
                                print(f"❌ Invalid class ID in {img_stem}.txt (line {line_num})")
                                print(f"   Class ID {class_id} not in range [0, {self.num_classes-1}]")
                                split_stats['invalid_format'] += 1
                                break
                            
                            # Validate coordinates (should be normalized 0-1)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                    0 <= width <= 1 and 0 <= height <= 1):
                                print(f"⚠️  Coordinates out of range in {img_stem}.txt (line {line_num})")
                                print(f"   Expected normalized values [0-1], got: {x_center}, {y_center}, {width}, {height}")
                            
                            objects_in_image += 1
                        
                        except ValueError as e:
                            print(f"❌ Value error in {img_stem}.txt (line {line_num}): {e}")
                            split_stats['invalid_format'] += 1
                            break
                
                if split_stats['invalid_format'] == 0:
                    split_stats['valid_images'] += 1
                    split_stats['total_objects'] += objects_in_image
                    split_stats['objects_per_image'].append(objects_in_image)
            
            except Exception as e:
                print(f"❌ Error reading {img_stem}.txt: {e}")
                split_stats['invalid_format'] += 1
        
        # Print summary for this split
        self._print_split_summary(split_stats)
        
        # Update total stats
        self.stats['total_images'] += split_stats['total_images']
        self.stats['total_labels'] += split_stats['total_labels']
        self.stats['images_without_labels'] += split_stats['images_without_labels']
        self.stats['empty_label_files'] += split_stats['empty_label_files']
        self.stats['invalid_format'] += split_stats['invalid_format']
        self.stats['valid_annotations'] += split_stats['valid_images']
        self.stats['total_objects'] += split_stats['total_objects']
        
        return len(missing_labels) == 0 and split_stats['invalid_format'] == 0
    
    def _print_split_summary(self, stats):
        """Print summary statistics for a split"""
        print(f"\n{'─'*70}")
        print(f"📋 {stats['split_name'].upper()} Split Summary:")
        print(f"{'─'*70}")
        print(f"Total images:              {stats['total_images']}")
        print(f"Total labels:              {stats['total_labels']}")
        print(f"Images without labels:     {stats['images_without_labels']} {'⚠️' if stats['images_without_labels'] > 0 else '✅'}")
        print(f"Empty label files:         {stats['empty_label_files']} {'⚠️' if stats['empty_label_files'] > 0 else '✅'}")
        print(f"Invalid format:            {stats['invalid_format']} {'⚠️' if stats['invalid_format'] > 0 else '✅'}")
        print(f"Valid images:              {stats['valid_images']}")
        
        if stats['total_objects'] > 0:
            avg_objects = stats['total_objects'] / stats['valid_images'] if stats['valid_images'] > 0 else 0
            min_objects = min(stats['objects_per_image']) if stats['objects_per_image'] else 0
            max_objects = max(stats['objects_per_image']) if stats['objects_per_image'] else 0
            print(f"Total objects:             {stats['total_objects']}")
            print(f"Avg objects per image:     {avg_objects:.2f}")
            print(f"Min/Max objects:           {min_objects}/{max_objects}")
        else:
            print(f"Total objects:             0 ❌ (NO ANNOTATIONS!)")
    
    def validate_all(self):
        """Validate all splits"""
        splits = ['train', 'val', 'test']
        all_valid = True
        
        for split in splits:
            split_path = self.dataset_path / split
            if split_path.exists():
                is_valid = self.validate_split(split)
                all_valid = all_valid and is_valid
        
        self._print_final_summary()
        return all_valid
    
    def _print_final_summary(self):
        """Print final overall summary"""
        print(f"\n\n{'='*70}")
        print("📊 OVERALL DATASET SUMMARY")
        print('='*70)
        print(f"Total images:              {self.stats['total_images']}")
        print(f"Total labels:              {self.stats['total_labels']}")
        print(f"Images without labels:     {self.stats['images_without_labels']} {'❌' if self.stats['images_without_labels'] > 0 else '✅'}")
        print(f"Empty label files:         {self.stats['empty_label_files']} {'❌' if self.stats['empty_label_files'] > 0 else '✅'}")
        print(f"Invalid format:            {self.stats['invalid_format']} {'❌' if self.stats['invalid_format'] > 0 else '✅'}")
        print(f"Valid images with labels:  {self.stats['valid_annotations']}")
        print(f"Total objects detected:    {self.stats['total_objects']} {'❌ (NO ANNOTATIONS!)' if self.stats['total_objects'] == 0 else '✅'}")
        print('='*70)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if self.stats['total_objects'] == 0:
            print("   ❌ CRITICAL: Your dataset has NO annotations!")
            print("   → This is why training fails (0 loss, 0 detections)")
            print("   → You need to annotate your images before training")
            print("   → Use a tool like Roboflow, LabelImg, or CVAT")
        elif self.stats['empty_label_files'] > 0:
            print(f"   ⚠️  {self.stats['empty_label_files']} label files are empty")
            print("   → These images have no objects (background images)")
            print("   → This is OK for detection, but verify intentionality")
        elif self.stats['images_without_labels'] > 0:
            print(f"   ⚠️  {self.stats['images_without_labels']} images have no label files")
            print("   → Either create labels or remove these images")
        else:
            print("   ✅ Dataset looks valid! Ready for training")
        
        if self.issues:
            print("\n📝 ISSUE LOG:")
            for issue_type, items in self.issues.items():
                if items:
                    print(f"\n   {issue_type}: {len(items)} issues")
                    for item in items[:5]:
                        print(f"      - {item}")
                    if len(items) > 5:
                        print(f"      ... and {len(items) - 5} more")


def main():
    """Main execution"""
    # Base path to dataset
    base_path = Path(r"D:\Seeker\Seeker - CV Project\YOLO Projects\Car Detector using YOLO26\Datasets\Military_Vehicles - Copy")
    
    if not base_path.exists():
        print(f"❌ Dataset path not found: {base_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🔍 YOLO DATASET VALIDATION TOOL")
    print("="*70)
    print(f"Dataset Path: {base_path}")
    
    # Create validator (6 classes as per your data.yaml)
    validator = DatasetValidator(base_path, num_classes=6)
    
    # Validate all splits
    is_valid = validator.validate_all()
    
    # Exit code based on validation result
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
