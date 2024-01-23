import numpy as np
import matplotlib.pyplot as plt
import os

# Function to plot the slices of the imaging and segmentation data
def plot_slices(imaging_path, segmentation_path):
    imaging_slice = np.load(imaging_path)
    segmentation_slice = np.load(segmentation_path)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(imaging_slice, cmap='gray')
    axes[0].set_title('Imaging Slice')
    axes[0].axis('off')

    axes[1].imshow(segmentation_slice[1], cmap='gray')
    axes[1].set_title('Segmentation Slice')
    axes[1].axis('off')
    plt.show()
    
def plot_slices(imaging_path, segmentation_path):
    imaging_slice = np.load(imaging_path)
    segmentation_slice = np.load(segmentation_path)

    # Squeeze the singleton dimension if it exists
    imaging_slice = np.squeeze(imaging_slice)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(imaging_slice, cmap='gray')
    axes[0].set_title('Imaging Slice')
    axes[0].axis('off')

    axes[1].imshow(segmentation_slice[1], cmap='gray')
    axes[1].set_title('Segmentation Slice')
    axes[1].axis('off')
    plt.show()


