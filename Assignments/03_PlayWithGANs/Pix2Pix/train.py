import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GeneratorNetwork import GeneratorNetwork
from DiscriminatorNetwork import DiscriminatorNetwork
from torch.optim.lr_scheduler import StepLR

IMAGE_GENERATION = True
LAMBDA = 100 # TODO

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator_model, discriminator_model, dataloader, gen_optimizer, dis_optimizer, bce_criterion, l1_criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        generator_model (nn.Module): The neural network generator model.
        discriminator_model (nn.Module): The neural network discriminator model.
        dataloader (DataLoader): DataLoader for the training data.
        gen_optimizer (Optimizer): Optimizer for updating generator model parameters.
        dis_optimizer (Optimizer): Optimizer for updating discriminator model parameters.
        bce_criterion (Loss): BCE loss function.
        l1_criterion (Loss): L1 loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator_model.train()
    discriminator_model.train()

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        if IMAGE_GENERATION:
            image_input, image_output = image_semantic, image_rgb
        else:
            image_input, image_output = image_rgb, image_semantic

        # Forward pass
        outputs = generator_model(image_input)

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_input, image_output, outputs, 'train_results', epoch)

        # Zero the gradients
        dis_optimizer.zero_grad()

        # Compute the loss
        real_p = discriminator_model(image_output, image_input)
        fake_p = discriminator_model(outputs.detach(), image_input)
        # dis_loss = torch.max(-(1.0000001 - fake_p).log(), -(real_p + 0.0000001).log()).sum()
        dis_loss = (bce_criterion(real_p, torch.ones_like(real_p)) + bce_criterion(fake_p, torch.zeros_like(fake_p))) / 2

        # Backward pass and optimization
        dis_loss.backward()
        dis_optimizer.step()
        # Print loss information
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Discriminator Loss: {dis_loss.item():g}, Real: {real_p.mean():g}, Fake: {fake_p.mean():g}')

        # Zero the gradients
        gen_optimizer.zero_grad()

        # Compute the loss
        fake_p = discriminator_model(outputs, image_input)
        if LAMBDA > 0:
            gen_loss = bce_criterion(fake_p, torch.ones_like(fake_p)) / LAMBDA + l1_criterion(outputs, image_output)
        else:
            gen_loss = bce_criterion(fake_p, torch.ones_like(fake_p))

        # Backward pass and optimization
        gen_loss.backward()
        gen_optimizer.step()
        # Print loss information
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Generator Loss: {gen_loss.item():.4f}')

def validate(generator_model, discriminator_model, dataloader, bce_criterion, l1_criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        generator_model (nn.Module): The neural network generator model.
        discriminator_model (nn.Module): The neural network discriminator model.
        dataloader (DataLoader): DataLoader for the validation data.
        bce_criterion (Loss): BCE loss function.
        l1_criterion (Loss): L1 loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator_model.eval()
    discriminator_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            if IMAGE_GENERATION:
                image_input, image_output = image_semantic, image_rgb
            else:
                image_input, image_output = image_rgb, image_semantic

            # Forward pass
            outputs = generator_model(image_input)

            # Compute the loss
            real_p = discriminator_model(image_output, image_input)
            fake_p = discriminator_model(outputs.detach(), image_input)
            # loss = bce_criterion(torch.cat((real_p, fake_p), dim=0), torch.cat((torch.ones_like(real_p), torch.zeros_like(fake_p)), dim=0))
            loss = l1_criterion(outputs, image_output)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_input, image_output, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Real: {real_p.mean():.4f}, Fake: {fake_p.mean():.4f}')


def get_recent_checkpoint():
    try:
        filenames = os.listdir('checkpoints')
        filenames.sort(key=lambda f: int(f[20:-4]))
        filename = filenames[-1]
        return int(filename[20:-4]), torch.load(os.path.join('checkpoints', filename), weights_only=True)
    except (FileNotFoundError, NotADirectoryError, IndexError):
        return 0, None


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=4)

    bce_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()

    # Initialize model, loss function
    generator_model = GeneratorNetwork().to(device)
    discriminator_model = DiscriminatorNetwork().to(device)

    # Load recent checkpoint if needed
    start_num_epochs, checkpoint = get_recent_checkpoint()
    if start_num_epochs > 0:
        generator_model.load_state_dict(checkpoint['generator_model_state_dict'])
        discriminator_model.load_state_dict(checkpoint['discriminator_model_state_dict'])
        # Initialize optimizers and add learning rate schedulers for decay
        gen_optimizer = optim.Adam([{'params': generator_model.parameters(), 'initial_lr': 0.0002}], lr=0.0002,
                                   betas=(0.5, 0.999))
        gen_scheduler = StepLR(gen_optimizer, step_size=200, gamma=0.2, last_epoch=start_num_epochs)
        dis_optimizer = optim.Adam([{'params': discriminator_model.parameters(), 'initial_lr': 0.0002}], lr=0.0002,
                                   betas=(0.5, 0.999))
        dis_scheduler = StepLR(dis_optimizer, step_size=200, gamma=0.2, last_epoch=start_num_epochs)
    else:
        if checkpoint:
            generator_model.load_state_dict(checkpoint)
        # Initialize optimizers and add learning rate schedulers for decay
        gen_optimizer = optim.Adam(generator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        gen_scheduler = StepLR(gen_optimizer, step_size=200, gamma=0.2)
        dis_optimizer = optim.Adam(discriminator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        dis_scheduler = StepLR(dis_optimizer, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(start_num_epochs, num_epochs):
        train_one_epoch(generator_model, discriminator_model, train_loader, gen_optimizer, dis_optimizer, bce_criterion,
                        l1_criterion, device, epoch, num_epochs)
        validate(generator_model, discriminator_model, val_loader, bce_criterion, l1_criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        gen_scheduler.step()
        dis_scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'generator_model_state_dict': generator_model.state_dict(),
                'discriminator_model_state_dict': discriminator_model.state_dict(),
            }, f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
