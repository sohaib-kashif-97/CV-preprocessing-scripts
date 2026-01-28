import sys
import os
from pathlib import Path

PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PATH, 'Vehicle Dataset')

def rename_image_label_pairs():
    """
    Rename image and label file pairs to have matching names.
    Renames files to sequential numbering (e.g., 001.jpg and 001.txt).
    """
    # Define the split directories (train and val; no 'test' split for now)
    splits = ['train', 'val']
    
    for split in splits:
        images_dir = os.path.join(DATASET_PATH, split, 'images')
        labels_dir = os.path.join(DATASET_PATH, split, 'labels')
        
        # Check if directories exist
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"⚠️  Warning: {split} split directories not found. Skipping...")
            continue
        
        # Get all image and label files in sorted order
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        label_files = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])
        
        # Verify same count
        if len(image_files) != len(label_files):
            print(f"❌ Error: {split} split has mismatched counts - {len(image_files)} images vs {len(label_files)} labels")
            continue
        
        print(f"\n📁 Processing {split} split ({len(image_files)} files)...")
        
        # Rename files with sequential numbering
        for idx, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
            # Get file extensions
            img_ext = os.path.splitext(img_file)[1]
            
            # Create new names with zero-padded numbers
            new_name = f"{idx:05d}"
            new_img_name = f"{new_name}{img_ext}"
            new_lbl_name = f"{new_name}.txt"
            
            # Full paths
            old_img_path = os.path.join(images_dir, img_file)
            old_lbl_path = os.path.join(labels_dir, lbl_file)
            new_img_path = os.path.join(images_dir, new_img_name)
            new_lbl_path = os.path.join(labels_dir, new_lbl_name)
            
            # Skip if already renamed correctly
            if img_file == new_img_name and lbl_file == new_lbl_name:
                print(f"  ✓ {idx + 1}/{len(image_files)} - Already named: {new_img_name}")
                continue
            
            # Rename image
            try:
                os.rename(old_img_path, new_img_path)
            except Exception as e:
                print(f"  ❌ Error renaming image {img_file}: {e}")
                continue
            
            # Rename label
            try:
                os.rename(old_lbl_path, new_lbl_path)
            except Exception as e:
                print(f"  ❌ Error renaming label {lbl_file}: {e}")
                # Try to revert image rename if label fails
                try:
                    os.rename(new_img_path, old_img_path)
                except:
                    pass
                continue
            
            print(f"  ✓ {idx + 1}/{len(image_files)} - Renamed to: {new_img_name} & {new_lbl_name}")
        
        print(f"✅ {split} split processing complete!")


def main():
    print('='*60)
    print('Renaming Image and Label File Pairs in Vehicle Dataset')
    print('='*60)
    
    try:
        rename_image_label_pairs()
        print('\n' + '='*60)
        print('✅ All files renamed successfully!')
        print('='*60)
    except Exception as e:
        print(f'\n❌ Fatal error: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()