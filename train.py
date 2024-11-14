# train.py

import torch
from tqdm import tqdm
import time
from test import *

def train_model(model, trainloader, valloader, criterion, optimizer, scheduler, 
                num_epochs=10, starting_epoch=0, device="cuda", seed=42):
    
    for epoch in range(starting_epoch, starting_epoch+num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        t1 = time.time()
        
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        t2 = time.time()
        per_epoch_time = t2-t1
        print(f"Took {per_epoch_time} seconds/epoch")

        avg_train_loss = running_loss / len(trainloader)
        val_loss, val_accuracy = evaluate_model_loss(model, valloader, criterion, "cuda")
        scheduler.step(val_loss)

        checkpoint_path = f"./vgg_epoch_{epoch+1}.pth"
        torch.save({
            'seed': seed,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'per_epoch_time': per_epoch_time
        }, checkpoint_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    return model
