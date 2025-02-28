# Import necessary libraries
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load HSI dataset
def load_dataset(dataset_name):
    if dataset_name == 'IN':
        data = sio.loadmat('./content/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = sio.loadmat('./content/Indian_pines_gt.mat')['indian_pines_gt']
    elif dataset_name == 'SV':
        data = sio.loadmat('./content/Salinas_corrected.mat')['salinas_corrected']
        gt = sio.loadmat('./content/Salinas_gt.mat')['salinas_gt']
    elif dataset_name == 'UP':
        data = sio.loadmat('./content/PaviaU.mat')['paviaU']
        gt = sio.loadmat('./content/PaviaU_gt.mat')['paviaU_gt']
    else:
        raise ValueError("Dataset not supported. Choose from 'IN', 'SV', or 'UP'.")
    
    return data, gt

# Preprocess data
def preprocess_data(data, gt, patch_length=4):
    # Normalize data
    data = MinMaxScaler().fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    
    # Pad data
    padded_data = np.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), mode='constant')
    
    # Do not flatten ground truth
    return padded_data, gt

# Create patches
def create_patches(data, gt, patch_length=4):
    patches, labels = [], []
    for i in range(patch_length, data.shape[0] - patch_length):
        for j in range(patch_length, data.shape[1] - patch_length):
            if gt[i - patch_length, j - patch_length] != 0:  # Ignore background
                patch = data[i-patch_length:i+patch_length+1, j-patch_length:j+patch_length+1, :]
                patches.append(patch)
                labels.append(gt[i - patch_length, j - patch_length] - 1)  # Subtract 1 to make labels start from 0
    return np.array(patches), np.array(labels)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=200):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),  # Adjust based on input patch size
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == y).sum().item()
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc/len(train_loader.dataset):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            val_loss += criterion(outputs, y).item()
            val_acc += (outputs.argmax(1) == y).sum().item()
    return val_loss / len(data_loader), val_acc / len(data_loader.dataset)

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    dataset_name = 'IN'  # Choose from 'IN', 'SV', or 'UP'
    data, gt = load_dataset(dataset_name)
    padded_data, gt = preprocess_data(data, gt)
    patches, labels = create_patches(padded_data, gt)
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(patches, labels, test_size=0.2, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: (batch, channels, H, W)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss, and optimizer
    num_classes = len(np.unique(labels))
    model = CNN(num_classes=num_classes, in_channels=data.shape[-1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=50)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
