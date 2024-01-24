import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd 
import torch.optim as optim
import ast
import csv
import time
from tqdm import tqdm
from tabulate import tabulate

# Assuming the correct device is set (e.g., "cuda" if a GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_triplets = 10  # Define the number of triplets to use for training (None means all triplets)
embedding_size = 2048  # Define the embedding size (must be the same as the output of the embedding layer)

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


# This class is to fine-tune the pre-trained ResNet152 model
class FineTuneResNet152(nn.Module):
    def __init__(self, feature_size, output_size):
        super(FineTuneResNet152, self).__init__()
        # Here we only use the fully connected layers for fine-tuning
        self.fc = nn.Linear(feature_size, output_size)

    def forward(self, x):
        # Directly use the embeddings for fine-tuning
        x = self.fc(x)
        return x
    
# Load the pre-trained ResNet152 model
original_model = models.resnet152(pretrained=True)

# Assume the output feature size of ResNet152 is 2048
model = FineTuneResNet152(feature_size=2048, output_size=2048).to(device)

# If you wish to fine-tune the entire network, set requires_grad to True
for param in model.parameters():
    param.requires_grad = True

# Define the optimizer for the fully connected layer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)


# Custom Dataset for loading embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, triplets_file, embeddings_folder, num_triplets=num_triplets):
        self.df_triplets = pd.read_csv(triplets_file)
        self.embeddings_folder = embeddings_folder
        self.files_in_memory = []
        self.image_embeddings = dict()
        # If num_triplets is provided, only keep that many triplets
        if num_triplets is not None:
            self.df_triplets = self.df_triplets[:num_triplets]

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
                temp = list(csv.DictReader(csvfile, delimiter=';'))

                for row in tqdm(temp, desc="Loading embeddings", total=len(temp)):
                    self.image_embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find file: {file_name}, please check if you have it downloaded')

        if str(image_id) in self.image_embeddings:
            return self.image_embeddings[str(image_id)]

        if str(image_id) not in self.image_embeddings:
            raise ValueError(f"Embedding of {image_id} not found. Please check {file_name} to see if it is included")



# Define transforms, dataset, and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Replace 'path_to_triplets.csv' and 'path_to_embeddings_folder' with your actual paths
dataset = EmbeddingDataset(triplets_file='Triplets.csv', embeddings_folder='Embedding Files', num_triplets=num_triplets)
# Define the sizes of train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

#length dataset
print('Length of dataset: ', len(dataset))
print('Length of train dataset: ', train_size)
print('Length of val dataset: ', val_size)
print('Length of test dataset: ', test_size)

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the dataloaders for train, validation, and test sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create a list of data
data = [
    ['Length of train dataloader:', len(train_dataloader)],
    ['Length of val dataloader:', len(val_dataloader)],
    ['Length of test dataloader:', len(test_dataloader)],
    ['Length of embeddings loaded:', len(dataset.image_embeddings)],
    ['Length of files that need to be loaded:', len(dataset.files_in_memory)]
]

# Print the table
print(tabulate(data, headers=['Data', 'Length']))



# Training loop
num_epochs = 20  # Define the number of epochs
epoch_losses = []
val_losses = []
epoch_times = []

#print min and max of train and val dataset
print('Min and max of train dataset: ', min(train_dataset.indices), max(train_dataset.indices))
print('Min and max of val dataset: ', min(val_dataset.indices), max(val_dataset.indices))

for epoch in range(num_epochs):
    start_time = time.time()  # Start time for the epoch
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_idx, (anchor, positive, negative) in progress_bar:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}')


    epoch_time = time.time() - start_time  # Time taken for the epoch
    epoch_times.append(epoch_time)

    # Calculate validation loss
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for val_batch_idx, (val_anchor, val_positive, val_negative) in enumerate(val_dataloader):
            val_anchor, val_positive, val_negative = val_anchor.to(device), val_positive.to(device), val_negative.to(device)

            val_anchor_output = model(val_anchor)
            val_positive_output = model(val_positive)
            val_negative_output = model(val_negative)

            val_loss += criterion(val_anchor_output, val_positive_output, val_negative_output).item()

    average_loss = total_loss / len(train_dataloader)
    val_average_loss = val_loss / len(val_dataloader)
    epoch_losses.append(average_loss)
    val_losses.append(val_average_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Val Loss: {val_average_loss:.4f}, Time: {epoch_time:.2f}s')

    # Save the model per epoch
    torch.save(model.state_dict(), f'resnet152_trained_triplet_epoch{epoch+1}.pt')

# Save the loss and time per epoch to a CSV file
with open('loss_and_time_per_epoch.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Time'])
    for epoch, train_loss, val_loss, time in zip(range(1, num_epochs+1), epoch_losses, val_losses, epoch_times):
        writer.writerow([epoch, train_loss, val_loss, time])


#table of loss and time per epoch without loading from csv
df = pd.DataFrame(list(zip(range(1, num_epochs+1), epoch_losses, val_losses, epoch_times))),
columns = ['Epoch', 'Train Loss', 'Val Loss', 'Time']

print(df)


# Calculate the minimum x and y values of the train and validation set
train_min_x = min(train_dataset.indices)
train_max_x = max(train_dataset.indices)
val_min_x = min(val_dataset.indices)
val_max_x = max(val_dataset.indices)

print('Min and max of train dataset: ', train_min_x, train_max_x)
print('Min and max of val dataset: ', val_min_x, val_max_x)

# Plot the time curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_times, marker='o', linestyle='-', color='blue')
plt.title('Training Time Curve')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)

# Add annotations for minimum x and y values
plt.annotate(f'Train Min: {train_min_x}', xy=(1, train_min_x), xytext=(1, train_min_x + 10),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate(f'Val Min: {val_min_x}', xy=(1, val_min_x), xytext=(1, val_min_x - 10),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

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
