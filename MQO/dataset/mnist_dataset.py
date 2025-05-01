import os
import struct
import torch
import pandas as pd
import numpy  as np
from array import array
import matplotlib.pyplot as plt

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir, split, transform=None):
        """
        Dataset class for loading the MNIST dataset.

        Args:
            data_dir (str): Path to data.
            split (str): train/test
            transform (callable, optional): Transformation to apply to the images.
        """
        self.data_dir            = data_dir
        self.split               = split.lower()
        self.transform           = transform
        self.images, self.labels = self.read_images_labels(data_dir)
    
    def read_images_labels(self, data_dir):
        """
        Reads the image and label files.

        Args:
            data_dir (str): Path to the data.

        Returns:
            images (list): List of 28x28 NumPy arrays representing the images.
            labels (list): List of labels.
        """
        training_images_filepath    = os.path.join(data_dir, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath    = os.path.join(data_dir, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath        = os.path.join(data_dir, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath        = os.path.join(data_dir, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        if(self.split == "train"):
            # Reading labels
            with open(training_labels_filepath, 'rb') as file:
                magic, size = struct.unpack('>II', file.read(8))
                if magic != 2049:
                    raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
                labels = array('B', file.read())        
            
            # Reading images
            with open(training_images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
                if magic != 2051:
                    raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
                image_data = array('B', file.read())

        if (self.split == "test"):  
            # Reading labels
            with open(test_labels_filepath, 'rb') as file:
                magic, size = struct.unpack('>II', file.read(8))
                if magic != 2049:
                    raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
                labels = array('B', file.read())        
            
            # Reading images
            with open(test_images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
                if magic != 2051:
                    raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
                image_data = array('B', file.read())      

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images.append(img)
                
        # Convert labels to NumPy array with int64 type
        labels = np.array(labels, dtype=np.int64)

        return images, labels

    def __len__(self):
        """Returns the total number of examples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a single example based on the given index.

        Args:
            idx (int): Index of the desired example.

        Returns:
            sample (dict): Dictionary containing 'image' and 'label'.
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)


        return image, label
    def save_image(self, idx, save_path):
        """
        Saves the image at the given index to the specified path.

        Args:
            idx (int): Index of the image to save.
            save_path (str): Path to save the image.
        """
        image = self.images[idx]
        plt.imsave(save_path, image, cmap='gray')
        print(f"Image at index {idx} saved to {save_path}.")

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])  
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long) 
    return images, labels




