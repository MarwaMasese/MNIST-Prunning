import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.cnn import CNN
from utils.train import train
from utils.evaluate import evaluate
from utils.prune import apply_pruning, remove_pruning

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer, device, epochs=3)

# Evaluate before pruning
print("\nModel evaluation before pruning:")
evaluate(model, test_loader, device)

# Apply pruning
apply_pruning(model, amount=0.5)

# Evaluate after pruning
print("\nModel evaluation after pruning:")
evaluate(model, test_loader, device)

# Remove pruning mask
remove_pruning(model)

# Final evaluation
print("\nModel evaluation after making pruning permanent:")
evaluate(model, test_loader, device)
