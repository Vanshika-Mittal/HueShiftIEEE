import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image
import cv2

class Dataset(Dataset):
    def __init__(self , root , transform=None):
        self.root = root
        self.files = os.listdir(root)
        self.len = len(self.files)
        if transform is not None:
            self.transforms = transforms.Compose(transform)
        else:
            self.transforms = None

        
    def __getitem__(self , i):
        file = self.files[i]
        im = cv2.imread(f'{self.root}/{file}')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        if self.transforms is not None:
            im = self.transforms(im)
        return im
    
    def __len__(self):
        return self.len
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    transform = [
        transforms.ToTensor(),
        transforms.Resize((64,64), Image.BICUBIC),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]


    data_loader = DataLoader(
        Dataset("data",transform, lab=True),
        batch_size = 1,
        shuffle = True,
        num_workers = 2
    )
    
    
    im, type = next(iter(data_loader))
    print(type)
    L,A,B = torch.split(im,[1,1,1],dim=1)

    print(L.shape)