import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("=== Improved X-Ray Model Training ===\n")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Enhanced data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

# Calculate class weights for balanced training
class_counts = np.bincount(train_dataset.targets)
class_weights = 1. / class_counts
sample_weights = class_weights[train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✓ Training images: {len(train_dataset)}")
print(f"✓ Validation images: {len(val_dataset)}")
print(f"✓ Classes: {train_dataset.classes}")
print(f"✓ Class distribution: Normal={class_counts[0]}, Pneumonia={class_counts[1]}\n")

# Model with better architecture
print("Setting up model...")
model = models.resnet18(pretrained=True)

# Freeze early layers
for param in list(model.parameters())[:-10]:
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

print("✓ Model ready\n")

# Training
num_epochs = 10
best_val_acc = 0
print(f"Starting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    train_acc = 100. * correct / total
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    normal_correct = 0
    normal_total = 0
    pneumonia_correct = 0
    pneumonia_total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]  ")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                if labels[i] == 0:  # Normal
                    normal_total += 1
                    if predicted[i] == 0:
                        normal_correct += 1
                else:  # Pneumonia
                    pneumonia_total += 1
                    if predicted[i] == 1:
                        pneumonia_correct += 1
            
            pbar.set_postfix({'acc': f'{100.*val_correct/val_total:.2f}%'})
    
    val_acc = 100. * val_correct / val_total
    normal_acc = 100. * normal_correct / normal_total if normal_total > 0 else 0
    pneumonia_acc = 100. * pneumonia_correct / pneumonia_total if pneumonia_total > 0 else 0
    
    print(f"\nEpoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    print(f"  Normal Acc={normal_acc:.2f}%, Pneumonia Acc={pneumonia_acc:.2f}%\n")
    
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'xray_model.pth')
        print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)\n")

print(f"=== TRAINING COMPLETE ===")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
