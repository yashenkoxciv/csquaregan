import os
import glob
import torch
import numpy as np
import csquaregan.logger as logger
from collections import defaultdict
from torch.utils import data
from skimage import io


class Dataset(data.Dataset):
    def __init__(self, image_dir, image_format='*.jpg', transform=None):
        image_dir = image_dir.rstrip('/')
        self.image_dir = image_dir
        root, image_dir_name = os.path.split(image_dir)
        self.kps_dir = os.path.join(root, 'kp_' + image_dir_name) 

        self.idx2images = defaultdict(list)
        for image_filename in glob.glob(os.path.join(os.path.normpath(image_dir), image_format)):
            _, image_name = os.path.split(image_filename)
            current_idx = image_name.split('_')[0]
            self.idx2images[current_idx].append(image_name)
        
        # integrity checking
        removed = 0
        self.image_count = 0
        idx_to_remove = []
        for idx in self.idx2images:
            current_images_count = len(self.idx2images[idx])
            if current_images_count < 2:
                idx_to_remove.append(idx)
                removed += 1
            else:
                self.image_count += current_images_count
        for idx in idx_to_remove:
            del self.idx2images[idx]
        if removed > 0:
            logger.warning('Integrity check.', removed, 'examples do not have pair. Deleted.')
        
        self.idxs = list(self.idx2images.keys())
        self.transform = transform
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        name1, name2 = np.random.choice(self.idx2images[self.idxs[index]], 2, False)

        i1 = np.array(io.imread(os.path.join(self.image_dir, name1)))
        k1 = np.array(io.imread(os.path.join(self.kps_dir, name1)))

        i2 = np.array(io.imread(os.path.join(self.image_dir, name2)))
        k2 = np.array(io.imread(os.path.join(self.kps_dir, name2)))

        if self.transform:
            i1 = self.transform(i1)
            i2 = self.transform(i2)
            k1 = self.transform(k1)
            k2 = self.transform(k2)

        return i1, i2, k1, k2

    def __str__(self):
        return f'<ReID-Dataset image-root={os.path.split(self.image_dir)[1]} kps-root={os.path.split(self.kps_dir)[1]} image-count={self.image_count} idxs={len(self.idx2images)}>'
    
    def __repr__(self):
        return str(self)
