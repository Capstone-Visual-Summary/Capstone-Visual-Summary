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
import pandas as pd 
import torch.optim as optim
import ast
import numpy as np

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()

# ResNet152 Embedding
class ResNet152Embedding(nn.Module):
    def __init__(self):
        super(ResNet152Embedding, self).__init__()
        resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet152.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x

# Custom Dataset for loading embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, triplets_file, embeddings_folder, transform=None):
        self.df_triplets = pd.read_csv(triplets_file)
        self.embeddings_folder = embeddings_folder
        self.transform = transform
        self.file_mapping = self.create_file_mapping()

    def create_file_mapping(self):
        # Create a mapping from a tuple of (lower_bound, upper_bound) to filename
        file_mapping = {}
        for filename in os.listdir(self.embeddings_folder):
            if filename.startswith('Embeddings_v1_') and filename.endswith('.csv'):
                parts = filename.replace('Embeddings_v1_', '').replace('.csv', '').split('_')
                if len(parts) == 2:
                    lower, upper = int(parts[0]), int(parts[1])
                    file_mapping[(lower, upper)] = filename
        return file_mapping

    def get_file_path(self, image_id):
        # Find the filename by checking which (lower_bound, upper_bound) tuple the image_id falls into
        for (lower, upper), filename in self.file_mapping.items():
            if lower <= image_id <= upper:
                return os.path.join(self.embeddings_folder, filename)
        return None

    def __getitem__(self, idx):
        anchor_id, positive_id, negative_id = self.df_triplets.iloc[idx]

        anchor_embedding = self.load_embedding(anchor_id)
        positive_embedding = self.load_embedding(positive_id)
        negative_embedding = self.load_embedding(negative_id)

        return anchor_embedding, positive_embedding, negative_embedding

    def __len__(self):
        return len(self.df_triplets)

    def load_embedding(self, image_id):
        file_name = self.get_file_path(image_id)
        if file_name is not None:
            embeddings_df = pd.read_csv(file_name, delimiter=';', header=None, skiprows=1)
            embedding_row = embeddings_df[embeddings_df[0] == str(image_id)].iloc[0, 1]
            embedding_list = ast.literal_eval(embedding_row)  # parse list-like string into actual list
            embedding = np.array(embedding_list, dtype=np.float32)  # convert list to numpy array
            return torch.from_numpy(embedding)  # convert numpy array to torch tensor
        else:
            # Handle the case where the file is not found
            print(f"Embedding file for image_id {image_id} not found.")

# Assuming the correct device is set (e.g., "cuda" if a GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms, dataset, and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Replace 'path_to_triplets.csv' and 'path_to_embeddings_folder' with your actual paths
dataset = EmbeddingDataset(triplets_file='Triplets.csv', embeddings_folder='Embedding Files', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and optimizer
resnet152 = ResNet152Embedding().to(device)
optimizer = optim.Adam(resnet152.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)

# Training loop
num_epochs = 20  # Define the number of epochs
epoch_losses = []

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
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}')

# Save the trained model
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