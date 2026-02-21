import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

print("=== Fast Efficient Training ===\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Minimal transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Smaller size = faster
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading balanced dataset...")
full_train = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

# Balance by undersampling
normal_idx = [i for i, l in enumerate(full_train.targets) if l == 0]
pneumonia_idx = [i for i, l in enumerate(full_train.targets) if l == 1]

random.seed(42)
pneumonia_idx = random.sample(pneumonia_idx, len(normal_idx))
balanced_idx = normal_idx + pneumonia_idx
random.shuffle(balanced_idx)

train_dataset = Subset(full_train, balanced_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # Larger batch
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

# Smaller, faster model
print("Setting up model...")
model = models.mobilenet_v2(pretrained=True)  # Faster than ResNet
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher LR = faster

print("✓ Ready\n")

epochs = 5  # Fewer epochs
best = 0

for epoch in range(epochs):
    # Train
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    # Validate
    model.eval()
    n_c, n_t, p_c, p_t = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).max(1)[1]
            
            for i in range(len(labels)):
                if labels[i] == 0:
                    n_t += 1
                    if pred[i] == 0: n_c += 1
                else:
                    p_t += 1
                    if pred[i] == 1: p_c += 1
    
    n_acc = 100. * n_c / n_t
    p_acc = 100. * p_c / p_t
    bal = (n_acc + p_acc) / 2
    
    print(f"Normal: {n_acc:.1f}%, Pneumonia: {p_acc:.1f}%, Balanced: {bal:.1f}%")
    
    if bal > best:
        best = bal
        torch.save(model.state_dict(), 'xray_model.pth')
        print("✓ Saved\n")
    else:
        print()

print(f"Done! Best: {best:.1f}%")
