import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A

from utils.helpers import generate_list_of_images
from utils import palette

def augmentations(image_size = 512):
    
    import albumentations as A
    return A.Compose([
        A.RandomCrop(width = image_size, height = image_size, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.Blur(p=0.01, blur_limit = 3),
    ], p = 1)


class MassuchusettsDataset(Dataset):
    

    def __init__(self,file_path, transforms = None, preprocessing = True):
        
        self.file_path = file_path
        self.transforms = transforms
        self.num_classes = 2
        self.preprocessing = preprocessing
        self.palette = palette.get_voc_palette(self.num_classes)
        self.MEAN = [0.4284687070054998, 0.4314829391019595, 0.39509143287278337]
        self.STD = [0.2920699753022457, 0.285627042080093, 0.2974879931626558]
        self.imgs, self.labels = self._prepare_data()
        
    def _prepare_data(self):
        return generate_list_of_images(self.file_path)
    
    def apply_normalization(self, img):
        
        transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize(
                            mean=self.MEAN,
                            std=self.STD)
                    ])
        
        return transform(img)
    
    def __len__(self):
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)
        
    def __getitem__(self, index):
        
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
        mask = cv2.imread(self.labels[index], -1)/255
        
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        if self.preprocessing:
            img = self.apply_normalization(img)
        
        return img, torch.Tensor(mask).long()
    