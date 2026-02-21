import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

print("=== Training for Perfect Accuracy ===\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Strong augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
full_train = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

# Use small balanced subset for perfect accuracy
normal_idx = [i for i, l in enumerate(full_train.targets) if l == 0]
pneumonia_idx = [i for i, l in enumerate(full_train.targets) if l == 1]

# Use only 200 images per class
random.seed(42)
normal_idx = random.sample(normal_idx, min(200, len(normal_idx)))
pneumonia_idx = random.sample(pneumonia_idx, min(200, len(pneumonia_idx)))

balanced_idx = normal_idx + pneumonia_idx
random.shuffle(balanced_idx)

train_dataset = Subset(full_train, balanced_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)} images (200 Normal + 200 Pneumonia)")
print(f"Val: {len(val_dataset)} images\n")

# Model
print("Setting up model...")
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("✓ Ready\n")

epochs = 20
best_balanced = 0
patience = 5
no_improve = 0

print(f"Training for up to {epochs} epochs...\n")

for epoch in range(epochs):
    # Train
    model.train()
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*train_correct/train_total:.1f}%'})
    
    train_acc = 100. * train_correct / train_total
    
    # Validate
    model.eval()
    n_c, n_t, p_c, p_t = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                if labels[i] == 0:
                    n_t += 1
                    if predicted[i] == 0: n_c += 1
                else:
                    p_t += 1
                    if predicted[i] == 1: p_c += 1
    
    n_acc = 100. * n_c / n_t if n_t > 0 else 0
    p_acc = 100. * p_c / p_t if p_t > 0 else 0
    bal = (n_acc + p_acc) / 2
    
    print(f"\nEpoch {epoch+1}:")
    print(f"  Train: {train_acc:.1f}%")
    print(f"  Normal: {n_acc:.1f}% ({n_c}/{n_t})")
    print(f"  Pneumonia: {p_acc:.1f}% ({p_c}/{p_t})")
    print(f"  Balanced: {bal:.1f}%")
    
    if bal > best_balanced:
        best_balanced = bal
        torch.save(model.state_dict(), 'xray_model.pth')
        print(f"  ✓ Best model saved!")
        no_improve = 0
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{patience})")
    
    print()
    
    # Early stopping if both classes reach 95%+
    if n_acc >= 95 and p_acc >= 95:
        print(f"Target accuracy reached! Stopping early.")
        break
    
    if no_improve >= patience:
        print(f"Early stopping triggered.")
        break

print(f"\n=== COMPLETE ===")
print(f"Best Balanced Accuracy: {best_balanced:.1f}%")
print(f"Model saved as 'xray_model.pth'")
