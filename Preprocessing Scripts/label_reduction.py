import os
import shutil

# ================= USER SETTINGS =================
DATASET_ROOT = r"D:\NASTP-K4\SEEKER_WORK\Object_detection_YOLO\Custom_Data\Tank_dataset\military_footage_recognition.v7i.yolov8"

# If True, original label files are backed up as .bak
BACKUP_LABELS = True
# =================================================

# Old YOLO class IDs
DELETE_CLASSES = {0, 1, 4}          # car, explosion, person
MERGE_TO_MILITARY = {2, 3, 5}       # military_truck, military_vehicle, truck

# New class ID
NEW_CLASS_ID = 0

# Common dataset split names
SPLITS = ["train", "val", "valid", "validation", "test"]


def process_label_file(label_path):
    """Rewrite a YOLO label file keeping only military vehicles."""
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        old_class = int(parts[0])

        if old_class in DELETE_CLASSES:
            continue

        if old_class in MERGE_TO_MILITARY:
            parts[0] = str(NEW_CLASS_ID)
            new_lines.append(" ".join(parts) + "\n")

    if BACKUP_LABELS:
        shutil.copy(label_path, label_path + ".bak")

    with open(label_path, "w") as f:
        f.writelines(new_lines)


def process_split(split_path):
    labels_dir = os.path.join(split_path, "labels")
    if not os.path.isdir(labels_dir):
        return

    print(f"Processing labels in: {labels_dir}")

    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, file)

        # ✅ Skip broken / missing files safely
        if not os.path.isfile(label_path):
            print(f"Skipping missing file: {label_path}")
            continue

        process_label_file(label_path)



def main():
    print("=== Reducing dataset to single class: military_vehicle ===")

    for split in SPLITS:
        split_path = os.path.join(DATASET_ROOT, split)
        if os.path.isdir(split_path):
            process_split(split_path)

    print("Done.")
    print("Final class mapping:")
    print("0 -> military_vehicle")


if __name__ == "__main__":
    main()
