from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import PIL
import os
import numpy as np



# data_transform = transforms.Compose([
#     transforms.Resize(size=(244, 244)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# im1 = Image.open("1.jpg")
# # print(im1.size())
# x = data_transform(im1)
# print(x)




class myDataSet(Dataset):
    def __init__(self, root, transform):
        self.image_files = np.array([x.path for x in os.scandir(root)
                                     if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.transform = transform

    def __getitem__(self, item):
        x = Image.open(self.image_files[item])
        x = self.transform(x)
        thisLabel =0
        if "dog" in self.image_files[item]:
            thisLabel = 1
        return x, thisLabel

    def __len__(self):
        return len(self.image_files)
    '''
    def __getitem__(self,index):
        img_path=self.imgs[index]
        if self.test:
            label=int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label=1 if 'dog' in img_path.split('/')[-1] else 0
        data=Image.open(img_path)
        data=self.transform(data)
        return data,label
    '''