from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CatDog_Dataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.len = len(imgs)
        self.transform = transform

    def __getitem__(self, index):
        imgs = np.random.permutation(self.imgs)
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transform(data)
        label = 0 if 'cat' in img_path.split('/')[-1] else 1
        return data, label

    def __len__(self):
        return len(self.imgs)
