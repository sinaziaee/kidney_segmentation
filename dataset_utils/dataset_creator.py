import torch
import os
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, kind='valid', root='/scratch/student/sinaziaee/home/src/kidney_segmentation', transform=None, normalize=False):
        """
        kind must be either 'train' or 'valid or test'
        """
        self.root = os.path.join(root, 'data_npy', kind)
        self.all_paths = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)] # cases
        self.images_path_list = []
        self.segmentation_path_list = []
        for case_path in sorted(self.all_paths): # for each case
            images_path = os.path.join(case_path, 'imaging') # imaging folder path of each case
            segmentation_path = os.path.join(case_path, 'segmentation') # segmentation folder path of each case
            for file_path in sorted(os.listdir(images_path)):
                self.images_path_list.append(os.path.join(images_path, file_path))
                self.segmentation_path_list.append(os.path.join(segmentation_path, file_path))
                
        self.transform = transform
        self.normalization = normalize
        if self.normalization:
            self.normalize = transforms.Normalize((0.5,), (0.5,))
    

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        # img_id = "{:04d}.npy".format(idx)
        # final_img_path = os.path.join(self.img_path, img_id.format(idx))
        # final_seg_path = os.path.join(self.seg_path, img_id.format(idx))
        final_img_path = self.images_path_list[idx]
        final_seg_path = self.segmentation_path_list[idx]
        
        img = torch.tensor(np.load(final_img_path), dtype=torch.float32)
        seg = torch.tensor(np.load(final_seg_path), dtype=torch.uint8)
        if self.normalization:
            img = self.normalize(img)

        if self.transform is not None:
            seed = np.random.randint(100)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)

            random.seed(seed)
            torch.manual_seed(seed)
            seg = self.transform(seg)

        return img, seg
    
    