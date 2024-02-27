import os
import random

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class Dataset2D(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
    
    @staticmethod
    def _load_input_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    @staticmethod
    def _load_target_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
            
    def __init__(self, input_root, target_root, transform=None, normalize=False, seed_fn=None):
        
        self.input_root = input_root
        self.target_root = target_root
        self.transform = transform
        self.seed_fn = seed_fn
        self.normalization = normalize
        if self.normalization:
            self.normalize = transforms.Normalize((0.5,), (0.5,))
                
        self.input_ids = sorted(img for img in os.listdir(self.input_root)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.target_ids = sorted(img for img in os.listdir(self.target_root)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        assert(len(self.input_ids) == len(self.target_ids))
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.seed_fn:
            self.seed_fn(seed)
        
    def __getitem__(self, idx):
        input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        target_img = self._load_target_image(
            os.path.join(self.target_root, self.target_ids[idx]))
        
        if self.normalization:
            input_img = self.normalize(input_img)
            
        if self.transform is not None:
            # transform should be the same for both image and target with the help of same seed
            seed = random.randint(0, 2**32)
            self._set_seed(seed)
            input_img = self.transform(input_img)
            self._set_seed(seed)
            target_img = self.transform(target_img)
        return input_img, target_img
        
    def __len__(self):
        return len(self.input_ids)