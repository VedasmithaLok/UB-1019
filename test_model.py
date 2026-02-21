import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

print("=== Testing Current Model ===\n")

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('xray_model.pth', map_location='cpu'))
model.eval()

print("✓ Model loaded\n")

# Load validation data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder('data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Testing on {len(val_dataset)} validation images...\n")

# Test
n_c, n_t, p_c, p_t = 0, 0, 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        for i in range(len(labels)):
            if labels[i] == 0:
                n_t += 1
                if predicted[i] == 0: n_c += 1
            else:
                p_t += 1
                if predicted[i] == 1: p_c += 1

n_acc = 100. * n_c / n_t
p_acc = 100. * p_c / p_t
overall = 100. * (n_c + p_c) / (n_t + p_t)
balanced = (n_acc + p_acc) / 2

print("=== RESULTS ===")
print(f"Normal Accuracy: {n_acc:.1f}% ({n_c}/{n_t})")
print(f"Pneumonia Accuracy: {p_acc:.1f}% ({p_c}/{p_t})")
print(f"Overall Accuracy: {overall:.1f}%")
print(f"Balanced Accuracy: {balanced:.1f}%")
print("\n" + "="*40)

if n_acc < 70 or p_acc < 70:
    print("⚠ Low accuracy detected!")
    print("Run: python train_perfect.py")
else:
    print("✓ Model performing well!")
