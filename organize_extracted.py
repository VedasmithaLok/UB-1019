import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== Organizing Extracted Dataset ===\n")

# Check if chest_xray folder exists
if not os.path.exists("chest_xray"):
    print("✗ chest_xray folder not found")
    exit()

print("✓ Found chest_xray folder\n")

# Create data structure
print("Creating data structure...")
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)

source_train = "chest_xray/train"
source_test = "chest_xray/test"

if os.path.exists(source_train):
    print("Copying training data...")
    for category in ["NORMAL", "PNEUMONIA"]:
        src = os.path.join(source_train, category)
        dst = os.path.join("data/train", category)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            count = len(os.listdir(dst))
            print(f"  ✓ Copied {count} {category} training images")
    
    print("\nCopying validation data...")
    for category in ["NORMAL", "PNEUMONIA"]:
        src = os.path.join(source_test, category)
        dst = os.path.join("data/val", category)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            count = len(os.listdir(dst))
            print(f"  ✓ Copied {count} {category} validation images")
    
    print("\n✓ Data organized successfully!")
    print("\nFinal structure:")
    print("data/")
    print("├── train/")
    print("│   ├── NORMAL/")
    print("│   └── PNEUMONIA/")
    print("└── val/")
    print("    ├── NORMAL/")
    print("    └── PNEUMONIA/")
else:
    print("✗ Dataset structure not found.")

print("\n=== DONE ===")
