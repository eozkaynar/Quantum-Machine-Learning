import os
import struct
import torch
import torchvision
import pandas as pd
import numpy  as np
from array import array
import matplotlib.pyplot as plt
import torchvision.transforms.functional

class MNISTDataset(torch.utils.data.Dataset):
    """
    Custom dataset for MNIST with support for quantum-specific preprocessing.

    Features:
    - Class filtering for binary/multiclass experiments.
    - Downsampling to lower resolutions (e.g., 4x4, 8x8) for quantum input.
    - Normalization to [0,1] gray scale.
    """
    def __init__(self,data_dir, split, selected_classes=None, transform=None):
        """
        Args:
            data_dir (str): Root directory containing MNIST files.
            split (str): 'train' or 'test'.
            selected_classes (list[int], optional): Only include samples with these labels.
            downsample_size (tuple[int, int], optional): If set, images will be downsampled to this resolution (e.g., (4, 4)).
            transform (callable, optional): Optional image transform (e.g., ToTensor, Normalize).
        """
        self.data_dir            = data_dir
        self.split               = split.lower()
        self.selected_classes    = selected_classes
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
        # Define file paths
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
    
    def filter_selected_classes(self):
        """Filters dataset to only include samples from selected_classes and remaps labels to 0..N."""
        filtered_images = []
        filtered_labels = []
        for img, label in zip(self.images, self.labels):
            if label in self.selected_classes:
                filtered_images.append(img)
                filtered_labels.append(self.selected_classes.index(label))  # re-label as 0, 1, ...
        self.images = filtered_images
        self.labels = np.array(filtered_labels, dtype=np.int64)

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




