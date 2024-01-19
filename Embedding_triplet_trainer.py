import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import random
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet152_Weights

num_epochs = 10  # Define the number of epochs
embedding_dimension = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()


class ResNet152Embedding(nn.Module):
    def __init__(self):
        super(ResNet152Embedding, self).__init__()
        resnet152 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # Keep all layers except the final fully connected layer
        self.features = nn.Sequential(*list(resnet152.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling to get [batch_size, 2048, 1, 1]
        x = torch.flatten(x, 1)  # Flatten the tensor to get [batch_size, 2048]
        return x

resnet152 = ResNet152Embedding()
resnet152 = resnet152.to(device)
    
# Define your transforms, dataset and dataloader here
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class StreetViewTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the locations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.locations = os.listdir(root_dir)
        self.images = {location: os.listdir(os.path.join(root_dir, location)) for location in self.locations}

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        location = self.locations[idx]
        location_images = self.images[location]

        anchor_img_name = random.choice(location_images)
        anchor_img_path = os.path.join(self.root_dir, location, anchor_img_name)

        # Get the year from the anchor image name
        anchor_year = int(anchor_img_name.split('_')[0])

        # Find positive image within 3 meters and in a different year
        positive_location = None
        positive_img_name = None
        while positive_location is None or positive_location == location or positive_img_name is None:
            positive_location = random.choice(self.locations)
            if abs(int(positive_location) - int(location)) <= 3:
                positive_images = self.images[positive_location]
                positive_img_name = random.choice(positive_images)
                positive_img_year = int(positive_img_name.split('_')[0])
                if positive_img_year == anchor_year:
                    positive_img_name = None

        positive_img_path = os.path.join(self.root_dir, positive_location, positive_img_name)

        # Find negative image within 100-500 meters and in the same year as the positive image
        negative_location = None
        negative_img_name = None
        while negative_location is None or negative_location == location or negative_img_name is None:
            negative_location = random.choice(self.locations)
            if abs(int(negative_location) - int(location)) >= 100 and abs(int(negative_location) - int(location)) <= 500:
                negative_images = self.images[negative_location]
                negative_img_name = random.choice(negative_images)
                negative_img_year = int(negative_img_name.split('_')[0])
                if negative_img_year != anchor_year:
                    negative_img_name = None

        negative_img_path = os.path.join(self.root_dir, negative_location, negative_img_name)

        # Load images
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

#dataset initialization
street_view_dataset = StreetViewTripletDataset(root_dir='/path/to/dataset', transform=transform)
dataloader = DataLoader(street_view_dataset, batch_size=32, shuffle=True)
# Assuming you have a dataloader 'dataloader' and the resnet152  resnet152'
optimizer = torch.optim.Adam(resnet152.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)

epoch_losses = []  # List to store loss of each epoch

for epoch in range(num_epochs):
    resnet152.train()
    total_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_embedding = resnet152(anchor)
        positive_embedding = resnet152(positive)
        negative_embedding = resnet152(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    epoch_losses.append(average_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}')

# Save the model
torch.save(resnet152.state_dict(), 'resnet152_trained_triplet.pt')

#plot the loss curve
# Assuming epoch_losses contains the average loss of each epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.show()