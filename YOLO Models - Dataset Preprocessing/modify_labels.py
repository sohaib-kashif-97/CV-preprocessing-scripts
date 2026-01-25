#!/usr/bin/env python3
"""
Dataset Label Fixing Script for YOLO Training
Handles:
1. Delete both .txt and .bak if both are empty + delete corresponding image
2. Keep .txt if it has valid content + delete .bak
3. Copy .bak content to .txt if .bak has content + delete .bak
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


class LabelFixer:
    def __init__(self, dataset_path, num_classes=1, dry_run=True):
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.dry_run = dry_run
        self.stats = {
            'bak_copied_to_txt': 0,
            'bak_empty_deleted': 0,
            'images_deleted': 0,
            'errors': 0
        }
        self.error_log = []
        self.operation_log = []
    
    def is_valid_yolo_format(self, content):
        """Check if content is valid YOLO format"""
        if not content or not content.strip():
            return False
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Check format: class_id x_center y_center width height
            if len(parts) != 5:
                return False
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate class ID
                if class_id < 0 or class_id >= self.num_classes:
                    return False
                
                # Validate coordinates (should be normalized 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    return False
            
            except ValueError:
                return False
        
        return True
    
    def get_file_content(self, file_path):
        """Read file content safely"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.error_log.append(f"Error reading {file_path}: {e}")
            return None
    
    def process_label_pair(self, images_dir, labels_dir, stem):
        """Process a pair of .txt and .bak files"""
        txt_path = labels_dir / f"{stem}.txt"
        bak_path = labels_dir / f"{stem}.bak"
        img_path = images_dir / f"{stem}.jpg"
        
        # Also check for other image extensions
        if not img_path.exists():
            for ext in ['.jpeg', '.png', '.bmp']:
                alt_path = images_dir / f"{stem}{ext}"
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        # Read .bak file
        bak_content = self.get_file_content(bak_path) if bak_path.exists() else ""
        
        # Case 1: .bak has valid content
        if self.is_valid_yolo_format(bak_content):
            self.operation_log.append(f"[COPY] .bak→.txt, delete .bak: {stem}")
            self.stats['bak_copied_to_txt'] += 1
            
            if not self.dry_run:
                try:
                    with open(txt_path, 'w') as f:
                        f.write(bak_content)
                    if bak_path.exists():
                        bak_path.unlink()
                except Exception as e:
                    self.error_log.append(f"Error processing {stem}: {e}")
                    self.stats['errors'] += 1
        
        # Case 2: .bak is empty or doesn't exist
        else:
            self.operation_log.append(f"[DELETE] .bak empty, delete .txt and image: {stem}")
            self.stats['bak_empty_deleted'] += 1
            
            if not self.dry_run:
                try:
                    if txt_path.exists():
                        txt_path.unlink()
                    if bak_path.exists():
                        bak_path.unlink()
                    if img_path.exists():
                        img_path.unlink()
                        self.stats['images_deleted'] += 1
                except Exception as e:
                    self.error_log.append(f"Error deleting {stem}: {e}")
                    self.stats['errors'] += 1
    
    def process_split(self, split_name):
        """Process a single split (train/val/test)"""
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} Split")
        print('='*70)
        
        images_dir = self.dataset_path / split_name / 'images'
        labels_dir = self.dataset_path / split_name / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  Split directories not found for {split_name}")
            return
        
        # Get all unique stems from label files
        label_files = list(labels_dir.glob('*.txt')) + list(labels_dir.glob('*.bak'))
        stems = set()
        for f in label_files:
            # Remove .txt or .bak extension
            stem = f.name.rsplit('.', 1)[0]
            stems.add(stem)
        
        print(f"Found {len(stems)} label pairs to process...")
        
        # Process each pair
        for i, stem in enumerate(sorted(stems), 1):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(stems)}...")
            self.process_label_pair(images_dir, labels_dir, stem)
        
        print(f"✅ Completed {split_name} split ({len(stems)} pairs processed)")
    
    def process_all(self):
        """Process all splits"""
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            split_path = self.dataset_path / split
            if split_path.exists():
                self.process_split(split)
    
    def print_summary(self):
        """Print summary statistics"""
        print(f"\n\n{'='*70}")
        print("📊 OPERATION SUMMARY")
        print('='*70)
        print(f"Status: {'DRY RUN (No changes made)' if self.dry_run else 'EXECUTED (Changes applied)'}")
        print(f"\n✅ .bak→.txt Copied:")
        print(f"   Files processed: {self.stats['bak_copied_to_txt']}")
        print(f"\n✅ .bak Empty → .txt & Image Deleted:")
        print(f"   Label pairs: {self.stats['bak_empty_deleted']}")
        print(f"   Images deleted: {self.stats['images_deleted']}")
        print(f"\n❌ Errors: {self.stats['errors']}")
        print('='*70)
        
        if self.error_log:
            print("\n⚠️  ERROR LOG:")
            for error in self.error_log[:10]:
                print(f"   {error}")
            if len(self.error_log) > 10:
                print(f"   ... and {len(self.error_log) - 10} more errors")


def main():
    """Main execution"""
    # Dataset path
    dataset_path = Path(r"D:\Seeker\Seeker - CV Project\YOLO Projects\Car Detector using YOLO26\Datasets\Military_Vehicles copy")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🔧 YOLO DATASET LABEL FIXER")
    print("="*70)
    print(f"Dataset Path: {dataset_path}")
    
    # Step 1: Dry run to preview changes
    print("\n" + "─"*70)
    print("STEP 1: DRY RUN (Preview all changes)")
    print("─"*70)
    
    fixer_dry = LabelFixer(dataset_path, num_classes=6, dry_run=True)
    fixer_dry.process_all()
    fixer_dry.print_summary()
    
    # Step 2: Ask for confirmation
    print("\n" + "─"*70)
    print("STEP 2: CONFIRMATION")
    print("─"*70)
    response = input("\n⚠️  Apply these changes? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Changes cancelled. Dataset not modified.")
        sys.exit(0)
    
    # Step 3: Execute changes
    print("\n" + "─"*70)
    print("STEP 3: EXECUTING CHANGES")
    print("─"*70)
    
    fixer_exec = LabelFixer(dataset_path, num_classes=6, dry_run=False)
    fixer_exec.process_all()
    fixer_exec.print_summary()
    
    print("\n✅ Dataset cleanup completed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
