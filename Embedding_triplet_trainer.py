import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset

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
        resnet152 = models.resnet152(pretrained=True)
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
# Example transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TripletStreetViewDataset(Dataset):
    # Initialize dataset with lists of anchor, positive, and negative image file paths
    def __init__(self, anchor_images, positive_images, negative_images, transform=None):
        self.anchor_images = anchor_images
        self.positive_images = positive_images
        self.negative_images = negative_images
        self.transform = transform

    def __getitem__(self, index):
        anchor = Image.open(self.anchor_images[index]).convert('RGB')
        positive = Image.open(self.positive_images[index]).convert('RGB')
        negative = Image.open(self.negative_images[index]).convert('RGB')
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative

    def __len__(self):
        return len(self.anchor_images)

#dataset initialization
dataset = TripletStreetViewDataset(anchor_images, positive_images, negative_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Assuming you have a dataloader 'dataloader' and the resnet152  resnet152'
optimizer = torch.optim.Adam(resnet152.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)

for epoch in range(num_epochs):
    resnet152.train()
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_embedding = resnet152(anchor)
        positive_embedding = resnet152(positive)
        negative_embedding = resnet152(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(resnet152.state_dict(), 'resnet152_trained_triplet.pt')