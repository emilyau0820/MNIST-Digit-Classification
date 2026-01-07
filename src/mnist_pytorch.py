# Project: MNIST Digit Classification
# File Name: mnist_pytorch.py
# Author: Emily Au

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

print(f"PyTorch: {torch.__version__}")

# 1. Data (auto-downloads MNIST)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 2. DataLoaders (PyTorch's tf.data equivalent)
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 3. Model (nn.Module)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet()

# 4. Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop (PyTorch explicit style)
EPOCHS = 5
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')

# 6. Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():  # No gradients for inference
    for x_batch, y_batch in test_loader:
        logits = model(x_batch)
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

test_acc = 100 * correct / total
print(f'Final Test Accuracy: {test_acc:.2f}%')

# 7. Save model
torch.save(model.state_dict(), 'mnist_pytorch.pt')
print("Model saved as 'mnist_pytorch.pt'")