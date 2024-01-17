import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, 2)
        negative_distance = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()


class ResNet152Embedding(nn.Module):
    def __init__(self, embedding_dimension=128):
        super(ResNet152Embedding, self).__init__()
        # Load a pre-trained ResNet152
        resnet152 = models.resnet152(pretrained=True)
        # Remove the last fully connected layer (classifier)
        self.features = nn.Sequential(*list(resnet152.children())[:-1])
        # Add a new fully connected layer for embeddings
        self.embedding = nn.Linear(resnet152.fc.in_features, embedding_dimension)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.embedding(x)
        return x
    

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


# Assuming you have a dataloader 'dataloader' and the model 'model'
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)

for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        optimizer.zero_grad()
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'resnet152_trained_triplet.pt')