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

# Vision Transformer (ViT) Model
class ViT(nn.Module):
    def __init__(self, image_size=9, patch_size=3, num_classes=16, dim=64, depth=6, heads=8, mlp_dim=128):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2 * data.shape[-1]  # Patch dimension
        
        self.patch_size = patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # Linear projection
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        # Extract patches
        p = self.patch_size
        patches = img.unfold(2, p, p).unfold(3, p, p)  # Shape: (batch, channels, H, W, p, p)
        patches = patches.contiguous().view(img.shape[0], -1, p * p * img.shape[1])  # Shape: (batch, num_patches, patch_dim)
        
        # Project patches to embedding space
        x = self.patch_to_embedding(patches)  # Shape: (batch, num_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)  # Shape: (batch, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch, num_patches + 1, dim)
        
        # Add positional embedding
        x += self.pos_embedding[:, :(patches.shape[1] + 1)]  # Shape: (batch, num_patches + 1, dim)
        
        # Pass through transformer
        x = self.transformer(x)  # Shape: (batch, num_patches + 1, dim)
        
        # Extract class token
        x = self.to_cls_token(x[:, 0])  # Shape: (batch, dim)
        
        # Pass through MLP head
        return self.mlp_head(x)  # Shape: (batch, num_classes)

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
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=20)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
