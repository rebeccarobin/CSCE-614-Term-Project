# vggTrain.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from train import *
from test import *

def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting seed of {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing for VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # VGG16 specific normalization (ImageNet values)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# dataset that is to be used : CIFAR 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# training, testing and validation set
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

# VGG16 typically uses smaller batch sizes due to memory requirements
batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize VGG16
model = models.vgg16(pretrained=False)
# Modify for CIFAR-10 (10 classes)
model.classifier[6] = nn.Linear(4096, 10)
model = model.to(device)

# Training parameters optimized for VGG16
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

load_model = True

if load_model == True:
    checkpoint = torch.load("./vgg_epoch_5.pth", map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    starting_epoch = checkpoint['epoch']
    seed = checkpoint['seed']
    set_seed(seed)
    num_epochs = 10
    start_time = time.time()
    trained_model = train_model(model, trainloader, valloader, criterion, optimizer, 
                              scheduler, num_epochs=num_epochs, starting_epoch=0, seed=seed)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds for {num_epochs} epochs")

else:
    seed = 42
    set_seed(seed)
    starting_epoch = 0
    num_epochs = 10
    start_time = time.time()
    trained_model = train_model(model, trainloader, valloader, criterion, optimizer, 
                              scheduler, num_epochs=num_epochs, starting_epoch=0, seed=seed)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds for {num_epochs} epochs")

test_loss, test_accuracy = evaluate_model_loss(model, testloader, criterion)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
