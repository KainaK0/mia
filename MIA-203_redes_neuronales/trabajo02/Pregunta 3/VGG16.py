import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Define the VGG16-like model
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv6
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 3
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Conv8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv9
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 4
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv11
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv12
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pool 5
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Adjust input size for 224x224 input images
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # Output layer
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc_layers(x)
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standards
])

# Load dataset from folder
data_dir = r"C:\Users\kainak0\Documents\gitProjects\mia\MIA-203_redes_neuronales\trabajo02\images_lung_cancer"  # Replace with your dataset path
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Split dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get the number of classes
num_classes = len(full_dataset.classes)

# Initialize the VGG16-like model
model = VGG16(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the model on the test images: {100 * correct / total}%')

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'cancer_vgg16m.pth')

# Function to display image with prediction
def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    if title is not None:
        plt.title(title, color='green')  # Green for correct, will change to red if incorrect
    plt.axis('off')
    plt.show()

# Get class names
class_names = full_dataset.classes

# Get a random image from the test set
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Select a single image
img = images[2000]
label = labels[2000]

# Perform inference
model.eval()
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()
    true_label = label.item()

# Create the title with true and predicted labels
title = f'True: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}'

# Display the image with labels
imshow(img, title)

# Print additional information
print(f'True label: {class_names[true_label]}')
print(f'Predicted label: {class_names[predicted_label]}')
if true_label == predicted_label:
    print('Prediction is correct!')
else:
    print('Prediction is incorrect.')

print('Finished Validation')