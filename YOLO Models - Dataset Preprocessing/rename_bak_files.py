#!/usr/bin/env python3
"""
Script to rename .bak label files back to .txt and remove old .txt label files
from the Military_Vehicles - Copy dataset.

This script performs the following operations:
1. Lists all .txt label files to be deleted
2. Asks for confirmation before deletion
3. Lists all .bak files to be renamed to .txt
4. Validates that no double .txt extensions exist in the filename
5. Verifies that corresponding image files exist for all label files
6. Performs the rename operation and shows confirmation
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Dataset paths
DATASET_BASE = r"d:\Seeker\Seeker - CV Project\YOLO Projects\Car Detector using YOLO26\Datasets\Military_Vehicles - Copy"
SUBDIRS = ['train', 'valid']  # Folders containing image and label subdirectories


def get_label_files(labels_dir: str) -> Tuple[List[str], List[str]]:
    """Get all .txt and .bak files in the labels directory."""
    txt_files = []
    bak_files = []
    
    if not os.path.exists(labels_dir):
        return txt_files, bak_files
    
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt.bak'):
            bak_files.append(filename)
        elif filename.endswith('.txt'):
            txt_files.append(filename)
    
    return sorted(txt_files), sorted(bak_files)


def get_image_files(images_dir: str) -> Set[str]:
    """Get all image files (jpg, png, etc.) in the images directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = set()
    
    if not os.path.exists(images_dir):
        return image_files
    
    for filename in os.listdir(images_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_files.add(os.path.splitext(filename)[0])  # Store without extension
    
    return image_files


def validate_no_double_extension(filename: str) -> bool:
    """Check that there are no multiple .txt extensions in the filename."""
    # Check the last 8 characters for double extensions
    if len(filename) < 8:
        return True
    
    last_8_chars = filename[-8:]
    if last_8_chars.count('.txt') > 1:
        return False
    
    return True


def get_label_base_name(label_file: str) -> str:
    """Extract the base name without any extension."""
    # Remove .txt or .txt.bak
    if label_file.endswith('.txt.bak'):
        return label_file[:-8]
    elif label_file.endswith('.txt'):
        return label_file[:-4]
    return label_file


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    while True:
        response = input(f"\n{message} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def main():
    """Main script execution."""
    
    print_section("Military Vehicles Label File Manager")
    print("This script will help you:")
    print("1. Remove old .txt label files")
    print("2. Rename .bak files to .txt files")
    print("3. Verify no file naming issues exist")
    print("4. Validate image-label file pairing")
    
    all_txt_to_delete = []
    all_bak_to_rename = []
    all_validation_issues = []
    
    # Process each subdirectory
    for subdir in SUBDIRS:
        labels_dir = os.path.join(DATASET_BASE, subdir, 'labels')
        images_dir = os.path.join(DATASET_BASE, subdir, 'images')
        
        txt_files, bak_files = get_label_files(labels_dir)
        image_files = get_image_files(images_dir)
        
        if not txt_files and not bak_files:
            print(f"⚠️  No label files found in {subdir}/labels")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {subdir}/")
        print(f"{'='*80}")
        
        # Step 1: Show .txt files to be deleted
        if txt_files:
            print(f"\n📋 Found {len(txt_files)} .txt files to be REMOVED:")
            print("-" * 80)
            for i, filename in enumerate(txt_files[:10], 1):  # Show first 10
                print(f"  {i:3d}. {filename}")
            if len(txt_files) > 10:
                print(f"  ... and {len(txt_files) - 10} more files")
            all_txt_to_delete.extend([(labels_dir, f) for f in txt_files])
        
        # Step 2: Show .bak files to be renamed
        if bak_files:
            print(f"\n📋 Found {len(bak_files)} .bak files to be RENAMED to .txt:")
            print("-" * 80)
            for i, filename in enumerate(bak_files[:10], 1):  # Show first 10
                new_name = filename[:-4]  # Remove .bak
                print(f"  {i:3d}. {filename}")
                print(f"       → {new_name}")
            if len(bak_files) > 10:
                print(f"  ... and {len(bak_files) - 10} more files")
            all_bak_to_rename.extend([(labels_dir, f) for f in bak_files])
        
        # Step 3: Validation
        print(f"\n🔍 Validation checks for {subdir}/:")
        print("-" * 80)
        
        validation_ok = True
        
        # Check for double .txt extensions in .bak files
        for bak_file in bak_files:
            if not validate_no_double_extension(bak_file):
                all_validation_issues.append((subdir, bak_file, "Double .txt extension detected"))
                print(f"  ❌ Double .txt extension: {bak_file}")
                validation_ok = False
        
        # Check if image files exist for renamed labels
        for bak_file in bak_files:
            label_base = get_label_base_name(bak_file)
            if label_base not in image_files:
                all_validation_issues.append((subdir, bak_file, "No matching image file found"))
                print(f"  ❌ No image file for label: {label_base}")
                validation_ok = False
        
        # Check if image files exist for current .txt files (will be deleted)
        for txt_file in txt_files:
            label_base = get_label_base_name(txt_file)
            if label_base not in image_files:
                # This is expected as old files will be deleted
                pass
        
        if validation_ok and (txt_files or bak_files):
            print(f"  ✅ All validations passed for {subdir}/")
        elif not txt_files and not bak_files:
            print(f"  ⓘ No files to process")
    
    # Summary
    print_section("Summary")
    print(f"Total .txt files to DELETE:  {len(all_txt_to_delete):4d}")
    print(f"Total .bak files to RENAME:  {len(all_bak_to_rename):4d}")
    print(f"Validation issues found:      {len(all_validation_issues):4d}")
    
    if all_validation_issues:
        print("\n⚠️  VALIDATION ISSUES DETECTED:")
        print("-" * 80)
        for subdir, filename, issue in all_validation_issues[:20]:
            print(f"  {subdir}: {filename}")
            print(f"    → {issue}")
        if len(all_validation_issues) > 20:
            print(f"  ... and {len(all_validation_issues) - 20} more issues")
        print("\n❌ Cannot proceed due to validation issues!")
        return False
    
    # Step 1: Confirm deletion of old .txt files
    print_section("Step 1: Delete Old .txt Label Files")
    
    if all_txt_to_delete:
        print(f"Ready to delete {len(all_txt_to_delete)} old .txt label files.")
        if confirm_action("Do you want to DELETE all old .txt files?"):
            deleted_count = 0
            for labels_dir, filename in all_txt_to_delete:
                try:
                    file_path = os.path.join(labels_dir, filename)
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"  ❌ Error deleting {filename}: {e}")
            
            print(f"\n✅ Successfully deleted {deleted_count}/{len(all_txt_to_delete)} files")
        else:
            print("❌ Deletion cancelled. Exiting...")
            return False
    else:
        print("No .txt files to delete.")
    
    # Step 2: Rename .bak files to .txt
    print_section("Step 2: Rename .bak Files to .txt")
    
    if all_bak_to_rename:
        print(f"Ready to rename {len(all_bak_to_rename)} .bak files to .txt")
        if confirm_action("Do you want to RENAME all .bak files to .txt?"):
            renamed_count = 0
            for labels_dir, filename in all_bak_to_rename:
                try:
                    old_path = os.path.join(labels_dir, filename)
                    new_filename = filename[:-4]  # Remove .bak
                    new_path = os.path.join(labels_dir, new_filename)
                    
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"  ❌ Error renaming {filename}: {e}")
            
            print(f"\n✅ Successfully renamed {renamed_count}/{len(all_bak_to_rename)} files")
        else:
            print("❌ Renaming cancelled. Exiting...")
            return False
    else:
        print("No .bak files to rename.")
    
    # Final verification
    print_section("Final Verification")
    
    for subdir in SUBDIRS:
        labels_dir = os.path.join(DATASET_BASE, subdir, 'labels')
        images_dir = os.path.join(DATASET_BASE, subdir, 'images')
        
        txt_files, bak_files = get_label_files(labels_dir)
        image_files = get_image_files(images_dir)
        
        print(f"\n{subdir}:")
        print(f"  Remaining .txt files:  {len(txt_files)}")
        print(f"  Remaining .bak files:  {len(bak_files)}")
        
        if bak_files == 0 and txt_files > 0:
            print(f"  ✅ Rename complete! Only .txt files remain.")
        
        # Verify image-label pairing
        unmatched_labels = 0
        for txt_file in txt_files:
            label_base = get_label_base_name(txt_file)
            if label_base not in image_files:
                unmatched_labels += 1
        
        if unmatched_labels > 0:
            print(f"  ⚠️  {unmatched_labels} labels without matching images")
        else:
            print(f"  ✅ All labels have matching images!")
    
    print_section("Process Complete!")
    print("✅ All operations completed successfully!")
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
