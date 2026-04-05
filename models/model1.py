import time
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

start_time = time.time()
# Define transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load the dataset
data_dir = "C:/Users/BHARAT/Desktop/data sets/image dataset/animals/image"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Get the class-to-index mapping
class_names = dataset.classes  # list of folder names (class names)
class_idx = {class_name: [] for class_name in class_names}

# Group indices by class
for idx, (_, label) in enumerate(dataset.samples):
    class_idx[class_names[label]].append(idx)

# Now split each class indices into 70% train and 30% validation
train_indices, val_indices = [], []

for class_name, indices in class_idx.items():
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)

# Create subsets for training and validation
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Check dataset size
print(f"Training size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# Define a simple CNN model for classification (adjust as needed)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(class_names)
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the parameters

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    print("Training complete.")

# Validation function
def validate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():  # No need to track gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            total_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Loss: {total_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Set the number of epochs
num_epochs = 10

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Validate the model
validate_model(model, val_loader)


end_time = time.time()
print(f'Total time: {end_time-start_time}')
