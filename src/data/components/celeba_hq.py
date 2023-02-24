from typing import Dict

import numpy as np
from torchvision.datasets import ImageFolder


class CelebAHQ(ImageFolder):
    def __init__(self, transform=None, train=True):
        suf = 'train' if train else 'val'
        super().__init__(f'data/celeba/celeba_hq/{suf}')
        self._transform = transform

    def __getitem__(self, item) -> Dict:
        image, y = super().__getitem__(item)
        image = np.array(image)
        if self._transform is not None:
            image = self._transform(image=image)['image']
        image = image / 255
        return {'image': image * 2 - 1, 'y': y}


class CelebAHQWithRefs(CelebAHQ):
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

