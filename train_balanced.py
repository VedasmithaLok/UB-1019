import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

print("=== Balanced X-Ray Model Training ===\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_counts = np.bincount(train_dataset.targets)
print(f"✓ Training images: {len(train_dataset)}")
print(f"✓ Validation images: {len(val_dataset)}")
print(f"✓ Classes: {train_dataset.classes}")
print(f"✓ Distribution: Normal={class_counts[0]}, Pneumonia={class_counts[1]}\n")

# Model
print("Setting up model...")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Use weighted loss for class imbalance
weight_ratio = float(class_counts[1]) / float(class_counts[0])
class_weights = torch.FloatTensor([weight_ratio, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("✓ Model ready with class balancing\n")

num_epochs = 8
best_val_acc = 0
best_balanced_acc = 0

print(f"Starting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # Training
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
    
    # Validation
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
            
            for i in range(len(labels)):
                if labels[i] == 0:
                    normal_total += 1
                    if predicted[i] == 0:
                        normal_correct += 1
                else:
                    pneumonia_total += 1
                    if predicted[i] == 1:
                        pneumonia_correct += 1
            
            pbar.set_postfix({'acc': f'{100.*val_correct/val_total:.2f}%'})
    
    val_acc = 100. * val_correct / val_total
    normal_acc = 100. * normal_correct / normal_total
    pneumonia_acc = 100. * pneumonia_correct / pneumonia_total
    balanced_acc = (normal_acc + pneumonia_acc) / 2
    
    print(f"\nEpoch {epoch + 1}:")
    print(f"  Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")
    print(f"  Normal Acc: {normal_acc:.2f}%")
    print(f"  Pneumonia Acc: {pneumonia_acc:.2f}%")
    print(f"  Balanced Acc: {balanced_acc:.2f}%\n")
    
    # Save best balanced model
    if balanced_acc > best_balanced_acc and normal_acc > 60:
        best_balanced_acc = balanced_acc
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'xray_model.pth')
        print(f"✓ Best model saved! (Balanced: {balanced_acc:.2f}%)\n")

print(f"=== TRAINING COMPLETE ===")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Best Balanced Accuracy: {best_balanced_acc:.2f}%")
