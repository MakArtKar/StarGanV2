import os

from PIL import Image

from src.data.components.celeba import CelebADataset


class WrappedCelebADataset(CelebADataset):
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[idx] # convert all attributes to zeros and ones
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(image=img)['image']
        idx = self.header.index('Male')
        return img, int(img_attributes[idx] == 1)
