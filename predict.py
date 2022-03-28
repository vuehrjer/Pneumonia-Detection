import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
torch.cuda.empty_cache()
from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = 'saved_models/'
model_name = 'pneumonia_model.pt'

cwd = os.getcwd()

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(cwd + '/data/chest_xray/test', transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

total_correct = 0
count = 0
classes = test_dataset.classes
file = torch.load(model_dir + model_name)
model = Net()
model.load_state_dict(file)
model.to(device)
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        yhat = model(images)
        y_pred = torch.round(yhat)
        eq = y_pred == labels.view(-1, 1)
        total_correct += eq.sum().item()

        if count % 20 == 0:
            print(f"Predicted {classes[int(y_pred.item())]}.. Ground Truth {classes[labels.item()]}..  {bool(eq)}")
        count += 1
print()
print()
print("TOTAL CORRECT: {}/{}".format(total_correct, len(test_dataset)))
print("PERCENTAGE CORRECT: {:.2f}%..".format((total_correct / len(test_dataset)) * 100))