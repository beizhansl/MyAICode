import os, glob
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

def getnoise(H,W):
    noise = np.zeros([H,W],dtype=np.uint8)
    noise = cv2.randn(noise,0,255)
    noise = np.asarray(noise/255,dtype=np.uint8)
    return noise

class SCFE(Dataset):
    def __init__(self, dataset_path, transform, mode, transform_mask):
        #C:\Users\zhans\Desktop\dataset(\celeba-512/sketch/color)
        self.dataset_path = dataset_path
        print("dataset path: ",self.dataset_path)
        self.transform = transform
        self.transform_mask = transform_mask
        self.mode = mode
        self.imgs = glob.glob(os.path.join(self.dataset_path, 'celeba-512', '*.jpg'))
        self.sketchs = glob.glob(os.path.join(self.dataset_path, 'celeba-sketch', '*.jpg'))
        self.colors = glob.glob(os.path.join(self.dataset_path, 'celeba-color', '*.jpg'))
        self.masks = glob.glob(os.path.join(self.dataset_path, 'celeba-mask', '*.jpg'))
        self.noimg = len(self.imgs)
        self.nomask = len(self.masks)
        print("The number of img: ",self.noimg)

    def __getitem__(self, index):
            idx = random.choice(range(self.noimg))
            #mask的数目是10000，更加随机一点
            idxmask = random.choice(range(self.nomask))
            img = Image.open(self.imgs[idx]).convert("RGB")
            sketch = Image.open(self.sketchs[idx])
            color = Image.open(self.colors[idx]).convert("RGB")
            mask = Image.open(self.masks[idxmask])
            noise = getnoise(512,512)
            #img, mask, sketch, color, noise
            return self.transform(img), self.transform_mask(mask), self.transform_mask(sketch), \
                    self.transform_mask(color),self.transform_mask(noise)

    def __len__(self):
        return self.noimg
