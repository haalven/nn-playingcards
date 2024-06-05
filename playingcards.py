#!/usr/bin/env python3

# playing cards classifier (neural network)


import torch, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib import pyplot
from PIL import Image
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
tests_folder = cards_folder + '/test'

# datasets
train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
tests_dataset = PlayingCardDataset(tests_folder, transform=transform)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
tests_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)



# the model - based on 'timm'
# https://github.com/huggingface/pytorch-image-models

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
num_epochs = 3

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
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

state_path = 'playingcards_trainstate.pt'
save_model(model, state_path)



# evaluation

# load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return image, transform(image).unsqueeze(0)

# predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# visualization
def visualize_predictions(original_image, probabilities, class_names):
    _, axarr = pyplot.subplots(1, 2, figsize=(14, 7))
    
    # display image
    axarr[0].imshow(original_image)
    axarr[0].axis('off')
    
    # display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel('probability')
    axarr[1].set_title('class predictions')
    axarr[1].set_xlim(0, 1)

    # show result
    pyplot.tight_layout()
    pyplot.show()

# the test image
test_image = tests_folder + '/queen of clubs/3.jpg'
original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, metal)
class_names = train_dataset.classes 
visualize_predictions(original_image, probabilities, class_names)

