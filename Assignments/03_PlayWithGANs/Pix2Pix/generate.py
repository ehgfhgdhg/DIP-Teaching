import os
import cv2
import numpy as np
import torch
import argparse
from DiscriminatorNetwork import DiscriminatorNetwork
from GeneratorNetwork import GeneratorNetwork

IMAGE_GENERATION = True
LAMBDA = 100

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


def get_recent_checkpoint():
    try:
        filenames = os.listdir('checkpoints')
        filenames.sort(key=lambda f: int(f[20:-4]))
        filename = filenames[-1]
        return int(filename[20:-4]), torch.load(os.path.join('checkpoints', filename), weights_only=True)
    except (FileNotFoundError, NotADirectoryError, IndexError):
        return 0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    input_folder, output_folder = args.input_folder, args.output_folder

    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss function, and optimizer
    generator_model = GeneratorNetwork().to(device)
    discriminator_model = DiscriminatorNetwork().to(device)

    l1_criterion = torch.nn.L1Loss()
    bce_criterion = torch.nn.BCELoss()

    # Load recent checkpoint if needed
    start_num_epochs, checkpoint = get_recent_checkpoint()
    if start_num_epochs > 0:
        generator_model.load_state_dict(checkpoint['generator_model_state_dict'])
        generator_model.eval()
        discriminator_model.load_state_dict(checkpoint['discriminator_model_state_dict'])
        discriminator_model.eval()
    else:
        raise RuntimeError("Cannot load recent model")

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        img_color_semantic = cv2.imread(os.path.join(input_folder, filename))
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        image_rgb = torch.Tensor(size=(1, image.size()[0], image.size()[1], image.size()[2]//2))
        image_semantic = torch.Tensor(size=(1, image.size()[0], image.size()[1], image.size()[2]//2))
        image_rgb[0, :, :, :] = image[:, :, :256]
        image_semantic[0, :, :, :] = image[:, :, 256:]

        with torch.no_grad():
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            if IMAGE_GENERATION:
                # Forward pass
                outputs = generator_model(image_semantic)

                # Compute the loss
                real_p = discriminator_model(image_rgb, image_semantic)
                fake_p = discriminator_model(outputs.detach(), image_semantic)
                loss = bce_criterion(fake_p, torch.ones_like(fake_p)) + l1_criterion(outputs, image_rgb) * LAMBDA

                # Save sample images
                # Convert tensors to images
                input_img_np = tensor_to_image(image_semantic[0])
                target_img_np = tensor_to_image(image_rgb[0])
                output_img_np = tensor_to_image(outputs[0])
            else:
                # Forward pass
                outputs = generator_model(image_rgb)

                # Compute the loss
                real_p = discriminator_model(image_semantic, image_rgb)
                fake_p = discriminator_model(outputs.detach(), image_rgb)
                loss = bce_criterion(fake_p, torch.ones_like(fake_p)) + l1_criterion(outputs, image_semantic) * LAMBDA

                # Save sample images
                # Convert tensors to images
                input_img_np = tensor_to_image(image_rgb[0])
                target_img_np = tensor_to_image(image_semantic[0])
                output_img_np = tensor_to_image(outputs[0])

            # Concatenate the images horizontally
            comparison = np.hstack((input_img_np, target_img_np, output_img_np))

            # Save the comparison image
            cv2.imwrite(os.path.join(output_folder, filename), comparison)

            print(f'"{filename}", loss={loss.item():.4f}, Real: {real_p.mean():.4f}, Fake: {fake_p.mean():.4f}')

if __name__ == '__main__':
    main()
