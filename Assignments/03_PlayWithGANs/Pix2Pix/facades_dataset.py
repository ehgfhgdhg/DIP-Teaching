import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        # Convert the image to a PyTorch tensor
        image_rgb = cv2.resize(img_color_semantic[:, :256, :], (286, 286))
        image_semantic = cv2.resize(img_color_semantic[:, 256:, :], (286, 286))

        rhw = torch.randint(0, 31, (2,))
        image_rgb, image_semantic = image_rgb[rhw[0]:rhw[0]+256, rhw[1]:rhw[1]+256, :], image_semantic[rhw[0]:rhw[0] + 256, rhw[1]:rhw[1] + 256, :]

        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_semantic = torch.from_numpy(image_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        return image_rgb, image_semantic