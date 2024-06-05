#!/usr/bin/env python3

# playing cards classifier (neural network) - training


import torch, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from os import environ


# datasets

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

# resize and transform images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# image paths
cards_folder = environ['HOME'] + '/Pictures/playing_cards'
train_folder = cards_folder + '/train'
valid_folder = cards_folder + '/valid'

# datasets
train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)



# the model - based on 'timm'

class SimpleCardClassifer(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCardClassifer, self).__init__()

        # where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = torch.nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        # make a classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output



# training

# epochs
num_epochs = 1

# load model to device
metal = torch.device('mps')  # Apple silicon hardware
model = SimpleCardClassifer(num_classes=53)  # 53 card types
model.to(metal)

# loss function
criterion = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# the training loop
train_losses, val_losses = [], []
for epoch in range(num_epochs):

    # training phase
    print(f'epoch {epoch+1}/{num_epochs} training phase…')
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # move inputs and labels to the device
        images, labels = images.to(metal), labels.to(metal)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # validation phase
    print(f'epoch {epoch+1}/{num_epochs} validation phase…')
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            # move inputs and labels to the device
            images, labels = images.to(metal), labels.to(metal)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(valid_loader.dataset)
    val_losses.append(val_loss)

    # report
    print(f'train loss: {train_loss}, val loss: {val_loss}')



# save training state

def save_model(model, path):
    print(f'writing state to {path}…')
    torch.save(model.state_dict(), path)

state_path = 'playingcards_trainstate.pt'
save_model(model, state_path)
