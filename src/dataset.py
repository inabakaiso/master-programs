import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur
)
from albumentations.pytorch.transforms import ToTensorV2

from nlp_process import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CaptionDataset(nn.Module):
    """
    Dataset
    """

    def __init__(self, df, tokenizer, transform=None):
        """
        :param df:
        :param img_dir:
        :param image_size:
        :param tokenizer:
        :param transform:
        """
        self.file_paths = df['file_name']
        self.captions = df['image_caption']
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Read image and process
        img = cv2.imread(os.path.join('../input/train_original/train2014', file_path))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        target = self.captions[idx]
        target = self.tokenizer.text_to_sequence(target)
        target_length = len(target)
        target_length = torch.LongTensor([target_length])
        return img, torch.LongTensor(target), target_length


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(224, 224),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# debug 用 => import pdb '\n' pdb.set_trace()
if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../input/cleaned_caption/clean_train.csv')
    tokenizer = torch.load('../model/tokenizer/tokenizer.pth')
    print(f"tokenizer stoi : {tokenizer.stoi}")

    ds = CaptionDataset(df, tokenizer, transform=get_transforms(data='train'))

    print(ds[0])



