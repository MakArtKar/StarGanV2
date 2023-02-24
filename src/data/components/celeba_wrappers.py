import os

import numpy as np
from PIL import Image

from src.data.components.celeba import CelebADataset


class WrappedCelebADataset(CelebADataset):
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[idx] # convert all attributes to zeros and ones
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        # Apply transformations to the image
        if self.transform:
            img = self.transform(image=img)['image']
            img = img / 255
        idx = self.header.index('Male')
        return {
            'image': img * 2 - 1,  # image \in [-1, 1]
            'y': int(img_attributes[idx] == 1),
        }


class WrappedCelebADatasetWithRefs(WrappedCelebADataset):
    def __init__(self, *args, num_refs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_refs = num_refs
        self.indices = []
        for _ in range(self.num_refs):
            self.indices.append(np.random.permutation(np.arange(len(self))))

    def __getitem__(self, item):
        result = super().__getitem__(item)
        for i in range(self.num_refs):
            dct = super().__getitem__(item)
            result.update({
                f'image_ref{i + 1}': dct['image'],
                f'y_ref{i + 1}': dct['y'],
            })
        return result
