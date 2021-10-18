from torchvision import transforms
import sys
import os

sys.path.append(os.path.join('../', 'src'))
import config

CFG = config.CFG()


def get_transforms(*, data):
    if data == 'train_original':
        return transforms.Compose([
            transforms.Resize(CFG.size, CFG.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Transpose(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.ToTensor(),
        ])
    elif data == "valid":
        return transforms.Compose([
            transforms.Resize(CFG.size, CFG.size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.ToTensor()
        ])
