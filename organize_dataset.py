import os
import zipfile
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== Manual Dataset Organizer ===\n")

# Check if zip file exists
zip_file = "chest-xray-pneumonia.zip"
if not os.path.exists(zip_file):
    print(f"✗ {zip_file} not found in current directory")
    print("\nPlease:")
    print("1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Click 'Download' button")
    print("3. Save the zip file to this folder")
    print("4. Run this script again")
    exit()

print(f"✓ Found {zip_file}\n")

# Extract dataset
print("Step 1: Extracting dataset...")
print("Extracting files (this may take a few minutes)...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    members = zip_ref.namelist()
    total = len(members)
    for i, member in enumerate(members, 1):
        zip_ref.extract(member, "dataset")
        if i % 500 == 0:
            print(f"  Extracted {i}/{total} files...")
print("✓ Extraction complete\n")

# Organize data structure
print("Step 2: Organizing data structure...")
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)

source_train = "dataset/chest_xray/train"
source_test = "dataset/chest_xray/test"

if os.path.exists(source_train):
    print("Copying training data...")
    for category in ["NORMAL", "PNEUMONIA"]:
        src = os.path.join(source_train, category)
        dst = os.path.join("data/train", category)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ Copied {category} training images")
    
    print("Copying validation data...")
    for category in ["NORMAL", "PNEUMONIA"]:
        src = os.path.join(source_test, category)
        dst = os.path.join("data/val", category)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ Copied {category} validation images")
    
    print("\n✓ Data organized successfully!")
    print("\nData structure:")
    print("data/")
    print("├── train/")
    print("│   ├── NORMAL/")
    print("│   └── PNEUMONIA/")
    print("└── val/")
    print("    ├── NORMAL/")
    print("    └── PNEUMONIA/")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree("dataset")
    os.remove(zip_file)
    print("✓ Cleanup complete!")
else:
    print("✗ Dataset structure not found.")

print("\n=== DONE ===")
