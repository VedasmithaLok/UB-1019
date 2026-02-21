import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import sys
import numpy as np
import random

sys.stdout.reconfigure(encoding='utf-8')

print("=== Properly Balanced Training ===\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Simple transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading and balancing dataset...")
full_train = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

# Balance training data by undersampling majority class
normal_indices = [i for i, label in enumerate(full_train.targets) if label == 0]
pneumonia_indices = [i for i, label in enumerate(full_train.targets) if label == 1]

print(f"Original: Normal={len(normal_indices)}, Pneumonia={len(pneumonia_indices)}")

# Undersample pneumonia to match normal count
random.seed(42)
pneumonia_indices = random.sample(pneumonia_indices, len(normal_indices))

balanced_indices = normal_indices + pneumonia_indices
random.shuffle(balanced_indices)

train_dataset = Subset(full_train, balanced_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Balanced: {len(train_dataset)} images (equal split)")
print(f"Validation: {len(val_dataset)} images\n")

# Simple model
print("Setting up model...")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("✓ Model ready\n")

num_epochs = 8
best_balanced = 0

print(f"Training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # Training
    model.train()
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
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    # Validation
    model.eval()
    normal_correct = 0
    normal_total = 0
    pneumonia_correct = 0
    pneumonia_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                if labels[i] == 0:
                    normal_total += 1
                    if predicted[i] == 0:
                        normal_correct += 1
                else:
                    pneumonia_total += 1
                    if predicted[i] == 1:
                        pneumonia_correct += 1
    
    normal_acc = 100. * normal_correct / normal_total
    pneumonia_acc = 100. * pneumonia_correct / pneumonia_total
    balanced_acc = (normal_acc + pneumonia_acc) / 2
    
    print(f"\nEpoch {epoch + 1}:")
    print(f"  Normal: {normal_acc:.1f}%")
    print(f"  Pneumonia: {pneumonia_acc:.1f}%")
    print(f"  Balanced: {balanced_acc:.1f}%")
    
    if balanced_acc > best_balanced:
        best_balanced = balanced_acc
        torch.save(model.state_dict(), 'xray_model.pth')
        print(f"  ✓ Saved!")
    print()

print(f"=== COMPLETE ===")
print(f"Best Balanced: {best_balanced:.1f}%")
