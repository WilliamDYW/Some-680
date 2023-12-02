from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import IAmodel

class ICCAD2019(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)  # Use 'train' or 'val' as split
        self.split = split
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, filename)
                    images.append((image_path, class_name))

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_name = self.images[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[class_name]
        return {'image': image, 'label': label}

# Example usage:
root_directory = '../2019/'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ICCAD2019(root_dir=root_directory, transform=transform)

train_dataset = ICCAD2019(root_dir=root_directory, split='train', transform=transform)
test_dataset = ICCAD2019(root_dir=root_directory, split='val', transform=transform)

# Create data loaders for training and testing
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your model, loss function, and optimizer
num_classes = len(dataset.classes)  # Adjust based on your dataset
model = IAmodel.SequentialInceptionAttentionBlocks(num_blocks=5)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    i = 0

    for batch in train_loader:
        print(i)
        i += 1

        inputs, labels = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        for batch in test_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            TP += (predicted == labels and predicted == torch.zeros(batch_size)).long().sum().item()
            TN += (predicted == labels and predicted == torch.ones(batch_size)).long().sum().item()
            FP += (predicted != labels and predicted == torch.zeros(batch_size)).long().sum().item()
            FN += (predicted != labels and predicted == torch.ones(batch_size)).long().sum().item()

        precision = TP / (TP+FP)
        recall    = TP / (TP+FN)
        print(f"Validation Loss: {val_loss / len(test_loader)}, Precision: {precision}, Recall: {recall}")

