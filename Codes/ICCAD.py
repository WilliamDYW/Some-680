from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import IAmodel
import BinRes

def normalize(data):
    if(data.data[0] < 0 and data.data[1] > 0):
        data.data[0], data.data[1] = 0,1
        return data
    if(data.data[0] > 0 and data.data[1] < 0):
        data.data[0], data.data[1] = 1,0
        return data
    data.data[0], data.data[1] = data.data[0]/(data.data[0]+data.data[1]),data.data[1]/(data.data[0]+data.data[1])
    return data


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
        image = Image.open(img_path).convert('RGB')

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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your model, loss function, and optimizer
num_classes = len(dataset.classes)  # Adjust based on your dataset
model = BinRes.ResNet_10(3)

print(torch.cuda.device_count())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

z = torch.zeros(batch_size)
o = torch.ones(batch_size)

z = z.to(device)
o = o.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
F1 = 0

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    i = 0

    for batch in train_loader:
        print(i, end=" ")
        i += 1

        inputs, labels = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        #for j in range(labels.size(0)):
        #   outputs[j,:] = normalize(outputs[j,:])
        #outputs.data = nn.functional.normalize(outputs.data, dim = -1)

        #print(outputs.data)	
        loss = criterion(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(running_loss)
    print()
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
        old_F1 = F1
        for batch in test_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            

            outputs = model(inputs)

            #for j in range(labels.size(0)):
            #    outputs[j,:] = normalize(outputs[j,:])

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            TP += ((predicted == labels).long() * (predicted == torch.zeros(labels.size(0)).to(device)).long()).sum().item()
            TN += ((predicted == labels).long() * (predicted == torch.ones(labels.size(0)).to(device)).long()).sum().item()
            FP += ((predicted != labels).long() * (predicted == torch.zeros(labels.size(0)).to(device)).long()).sum().item()
            FN += ((predicted != labels).long() * (predicted == torch.ones(labels.size(0)).to(device)).long()).sum().item()
            if(TP + FP == 0):
                precision = 0
            else:
                precision = TP/(TP+FP)
            if(TP + FN == 0):
                recall = 0
            else:
                recall = TP/(TP+FN)
            if(precision + recall == 0):
                F1 = 0
            else:
                F1 = 2*precision*recall/(precision+recall)
            if(old_F1 < F1):
                torch.save(model.state_dict(), "./whole_best.pt")
                print("saved!")



        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Validation Loss: {val_loss / len(test_loader)}, precision: {precision}, recall: {recall}, F1: {F1}")
