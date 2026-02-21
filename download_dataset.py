import os
import zipfile
import shutil
import sys

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

print("=== X-Ray Dataset Downloader ===\n")

# Step 1: Install dependencies
print("Step 1: Installing dependencies...")
os.system("pip install kaggle tqdm -q")
print("✓ Dependencies installed\n")

# Step 2: Verify Kaggle setup
print("Step 2: Verifying Kaggle API setup...")
kaggle_path = f"C:\\Users\\{os.getenv('USERNAME')}\\.kaggle\\kaggle.json"
if os.path.exists(kaggle_path):
    print(f"✓ Found kaggle.json at {kaggle_path}\n")
else:
    print(f"✗ kaggle.json not found at {kaggle_path}")
    print("Please ensure the file exists and try again.")
    exit()

# Step 3: Download dataset with retry
print("Step 3: Downloading dataset (1.2GB - may take 5-15 minutes)...")
dataset = "paultimothymooney/chest-xray-pneumonia"

max_retries = 3
for attempt in range(1, max_retries + 1):
    print(f"\nAttempt {attempt}/{max_retries}...")
    result = os.system(f"kaggle datasets download -d {dataset}")
    
    if result == 0 and os.path.exists("chest-xray-pneumonia.zip"):
        print("✓ Download completed successfully!")
        break
    else:
        print(f"✗ Attempt {attempt} failed")
        if attempt < max_retries:
            print("Retrying...")
        else:
            print("\n✗ Download failed after 3 attempts.")
            print("Try: 1) Check internet connection 2) Download manually from Kaggle")
            exit()

# Step 4: Extract dataset
print("\nStep 4: Extracting dataset...")
zip_file = "chest-xray-pneumonia.zip"
if os.path.exists(zip_file):
    print("Extracting files (this may take a few minutes)...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        members = zip_ref.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zip_ref.extract(member, "dataset")
            if i % 500 == 0:
                print(f"  Extracted {i}/{total} files...")
    os.remove(zip_file)
    print("✓ Dataset extracted to 'dataset' folder")
else:
    print("✗ Zip file not found.")
    exit()

# Step 5: Organize data structure
print("\nStep 5: Organizing data structure...")
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)

source_train = "dataset/chest_xray/train"
source_val = "dataset/chest_xray/val"
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
    print("✓ Cleanup complete!")
    print("\nNote: This dataset has 2 classes (Normal, Pneumonia).")
    print("You'll need to update model.py to use 2 classes instead of 5.")
else:
    print("✗ Dataset structure not found.")

print("\n=== DONE ===")
