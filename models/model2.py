import time
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Check dataset size
print(f"Training size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer with a layer that matches your number of classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
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

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():  # No need to track gradients during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Collect all labels and predictions for classification report
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # Generate classification report
        class_report = classification_report(all_labels, all_predictions, target_names=class_names)

        # Print metrics for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print("Classification Report:")
        print(class_report)

    print("Training complete.")

# Set the number of epochs
num_epochs = 10

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

end_time = time.time()
print(f'Total time: {end_time-start_time:.2f} seconds')

# Save the trained model
torch.save(model.state_dict(), 'resnet18_model.pth')
