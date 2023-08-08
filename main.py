import os
import argparse
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from unet.unet_model import UNet  
from utils.dice_loss import dice_loss
from utils.custom_dataset import CustomDataset
from utils.preprocessing import foreground
from utils.train_val_split import train_val_split


def train_unet(args):

    data_folder = args.dataset_path
    foreground(data_folder)
    train_ids, val_ids = train_val_split(data_folder, train_size=args.train_size)

    # Initialize U-Net model
    model = UNet(n_channels = 3 , n_classes = 1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Prepare the datasets and data loaders
    train_dataset = CustomDataset(data_folder, image_ids=train_ids, mask_ids=train_ids, transform=data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CustomDataset(data_folder, image_ids=val_ids, mask_ids=val_ids, transform=data_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize wandb

    wandb.init(project="pytorch_unet", config={"learning_rate": args.lr, "batch_size": args.batch_size})
    wandb.config.epochs = args.epoch
    wandb.config.model_architecture = "UNet"

    # Training loop
    for epoch in range(args.epoch):

        # Train the model
        model.train()
        for inputs, targets in train_dataloader:

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = dice_loss(outputs, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training loss on wandb
        wandb.log({"train_loss": loss.item()})

        # Print training statistics 
        print(f"Epoch [{epoch+1}/{2}], Train_Loss: {loss.item():.4f}")

        # Validate the model
        model.eval()  
        with torch.no_grad():

            for inputs_val, targets_val in val_dataloader:
                outputs_val = model(inputs_val)
                loss_val = dice_loss(outputs_val, targets_val)

        # Log validation loss on wandb
        wandb.log({"val_loss": loss_val.item()})

        # Print validation statistics 
        print(f"Epoch [{epoch+1}/{2}], Val_Loss: {loss_val.item():.4f}")
                

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)
    print("Training completed and model saved successfully!")

    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Model Training")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("train_size", type=float, help="Train size for train and validation split")
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("epoch", type=int, help="Epochs")
    parser.add_argument("model_save_path", type=str, help="Path to model save folder")


    args = parser.parse_args()

    train_unet(args)
