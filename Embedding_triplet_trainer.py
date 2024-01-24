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
import csv
import time
from sklearn.model_selection import train_test_split


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
        self.files_in_memory = []
        self.image_embeddings = dict()

    def get_file_path(self, version, img_id_com):
        version_split = str(version).split('.')
        version_str = f'v{version_split[0]}_{version_split[1]}'

        files = ['Embedding Files/' + f for f in os.listdir('Embedding Files') if version_str in f]
        files = sorted(files, key=lambda x: int(x.split('_')[3].split('_')[0]))

        return files[int(img_id_com) // 2392]

    def __getitem__(self, idx):
        anchor_id, positive_id, negative_id = self.df_triplets.iloc[idx]

        anchor_embedding = self.load_embedding(anchor_id)
        positive_embedding = self.load_embedding(positive_id)
        negative_embedding = self.load_embedding(negative_id)

        return anchor_embedding, positive_embedding, negative_embedding

    def __len__(self):
        return len(self.df_triplets)

    def load_embedding(self, image_id):
        file_name = self.get_file_path(1.0, image_id)

        if len(self.files_in_memory) == 100:
            lower_bound = int(self.files_in_memory[0].split('_')[3].split('_')[0])
            upper_bound = int(self.files_in_memory[0].split('_')[4].split('.')[0]) + 1

            for i in range(lower_bound, upper_bound):
                del self.image_embeddings[int(i)]

            del self.files_in_memory[0]

            self.files_in_memory.append(file_name)
        else:
            self.files_in_memory.append(file_name)

        try:
            with open(file_name, mode='r', newline='', encoding='utf-8') as csvfile:
                temp = csv.DictReader(csvfile, delimiter=';')

                for row in temp:
                    self.image_embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find file: {file_name}, please check if you have it downloaded')

        if str(image_id) in self.image_embeddings:
            return self.image_embeddings[str(image_id)]

        if str(image_id) not in self.image_embeddings:
            raise ValueError(f"Embedding of {image_id} not found. Please check {file_name} to see if it is included")

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
# Define the sizes of train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the dataloaders for train, validation, and test sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model and optimizer
resnet152 = ResNet152Embedding().to(device)
optimizer = optim.Adam(resnet152.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)

# Training loop
num_epochs = 20  # Define the number of epochs
epoch_losses = []
val_losses = []
epoch_times = []

for epoch in range(num_epochs):
    resnet152.train()
    total_loss = 0
    
    start_time = time.time()  # Start time of the epoch

    for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_embedding = resnet152(anchor)
        positive_embedding = resnet152(positive)
        negative_embedding = resnet152(negative)

        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.time() - start_time  # Time taken for the epoch
    epoch_times.append(epoch_time)

    # Calculate validation loss
    resnet152.eval()
    val_loss = 0

    with torch.no_grad():
        for val_batch_idx, (val_anchor, val_positive, val_negative) in enumerate(val_dataloader):
            val_anchor, val_positive, val_negative = val_anchor.to(device), val_positive.to(device), val_negative.to(device)

            val_anchor_embedding = resnet152(val_anchor)
            val_positive_embedding = resnet152(val_positive)
            val_negative_embedding = resnet152(val_negative)

            val_loss += criterion(val_anchor_embedding, val_positive_embedding, val_negative_embedding).item()

    average_loss = total_loss / len(train_dataloader)
    val_average_loss = val_loss / len(val_dataloader)
    epoch_losses.append(average_loss)
    val_losses.append(val_average_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Val Loss: {val_average_loss:.4f}, Time: {epoch_time:.2f}s')

    # Save the model per epoch
    torch.save(resnet152.state_dict(), f'resnet152_trained_triplet_epoch{epoch+1}.pt')

# Save the loss and time per epoch to a CSV file
with open('loss_and_time_per_epoch.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Time'])
    for epoch, train_loss, val_loss, time in zip(range(1, num_epochs+1), epoch_losses, val_losses, epoch_times):
        writer.writerow([epoch, train_loss, val_loss, time])

# Plot the loss curve
# Assuming epoch_losses contains the average train loss of each epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='blue', label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='o', linestyle='-', color='red', label='Val Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.grid(True)
plt.show()

#plot the time curve
# Assuming epoch_times contains the time of each epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_times, marker='o', linestyle='-', color='blue')
plt.title('Training Time Curve')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.show()
