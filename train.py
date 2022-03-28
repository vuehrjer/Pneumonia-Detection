import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
from torchsummary import summary

import wandb

# wandb.init(project="my-test-project", entity="riyuma")

torch.cuda.empty_cache()
from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

model_dir = 'saved_models/'
model_name = 'pneumonia_model.pt'

cwd = os.getcwd()

train_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomRotation(degrees=(-20, +20)),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(cwd + '/data/chest_xray/train', transform=train_transform)
valid_dataset = ImageFolder(cwd + '/data/chest_xray/val', transform=valid_transform)
test_dataset = ImageFolder(cwd + '/data/chest_xray/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=35, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

images, labels = iter(train_loader).next()

#summary(Net().cuda(), (images.shape[1], images.shape[2], images.shape[3]))

"""plt.figure(figsize = (20, 20))
for i in range(5):
    image = transforms.ToPILImage()(train_dataset[i][0])
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
plt.show()"""



def train(model, epochs, criterion, optimizer):
    for epoch in range(epochs):
        training_loss = 0
        model.train()
        i = 1
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            yhat = model(images)
            labels = labels.unsqueeze(1)
            labels = labels.float()
            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
            # wandb.log({'epoch': epoch, 'loss': training_loss})
            if i % 10 == 0:
                print("\tEPOCH: {}.. TRAINING LOSS: {:.6f}".format(epoch + 1, training_loss / 10))
                training_loss = 0.0
            i += 1
            # wandb.watch(model, criterion, log="all")
            del images, labels
            torch.cuda.empty_cache()

        model.eval()
        valid_loss, acc = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                yhat = model(images)
                labels = labels.unsqueeze(1)
                labels = labels.float()
                loss = criterion(yhat, labels)

                y_pred = torch.round(yhat)
                eq = y_pred == labels.view(-1, 1)
                acc += eq.sum().item()

                valid_loss += loss.item()

                del images, labels
                torch.cuda.empty_cache()

        acc = (acc / len(valid_dataset)) * 100
        print("EPOCH: {}/{}.. \tTRAINING LOSS: {:.6f}.. \tVALID LOSS: {:.6f}.. \tACCURACY: {:.2f}%..".format(epoch + 1,
                                                                                                             epochs,
                                                                                                             training_loss,
                                                                                                             valid_loss,

                                                                                                             acc))
        torch.save(model.state_dict(), model_dir + model_name)



def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)


model = Net()

try:
    file = torch.load(model_dir + model_name)
    model.load_state_dict(file)
except Exception:
    print("model file not found or state dict does not fit model architecture")
    #model.apply(weights_init_uniform)
criterion = nn.BCELoss()
EPOCHS = 30
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
model.to(device)
train(model, EPOCHS, criterion, optimizer)
