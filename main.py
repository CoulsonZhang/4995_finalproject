import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

import sys
from tqdm import tqdm
import time
import copy



from dataset import *
from model import *
from file import *
from preprocessing import *


from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
from torch.utils.data import *

import albumentations as A
import torch, cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F
from timeit import default_timer as timer

fold = 1
train_batch_size = 8
valid_batch_size = 16
start_lr   = 1e-4
num_iteration = 12000
iter_log    = 200
iter_valid  = 200


def create_siim_dataloader(meta_csv_path='./jpg_form/meta.csv'):
    jpg_df = pd.read_csv(meta_csv_path)
    train_df = jpg_df.loc[jpg_df['split'] == 'train']

    train_transform = A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0),
        A.ShiftScaleRotate(),
        A.GlassBlur(),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225]),
        ToTensorV2()
    ])
    df_train, df_valid = make_fold(mode='train', fold=fold)
    train_dataset = SiimDataset(df_train, transform=train_transform)
    valid_dataset = SiimDataset(df_valid)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=valid_batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader


dataset_path = "./butterfly-images40-species"

(train_loader, train_data_len) = get_data_loaders(dataset_path, 128, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)

print(train_data_len)
examples = next(iter(train_loader))

for label, img  in enumerate(examples[0]):
   plt.imshow(img.permute(1,2,0))
   plt.show()
   print(f"Label: {label}")