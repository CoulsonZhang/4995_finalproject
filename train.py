from dataset import *
from model import *
from file import *
from preprocessing import *
# from swin import *


from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from madgrad import MADGRAD

import torch, cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F
from timeit import default_timer as timer

import warnings

warnings.filterwarnings("ignore")
    
    
    
fold = 1
train_batch_size = 4
valid_batch_size = 4
start_lr   = 1e-4
num_iteration = 12000
iter_log    = 10
iter_valid  = 100

best_acc = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device={device}')
log = Logger()
log.open('./log.train.txt', mode='a+')

def create_siim_dataloader(meta_csv_path='./jpg_form/meta.csv'):
    jpg_df = pd.read_csv(meta_csv_path)
    train_df = jpg_df.loc[jpg_df['split'] == 'train']
    
    train_transform = A.Compose([
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0),
        A.ShiftScaleRotate(),
        A.GlassBlur(),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5],std=[0.225, 0.225, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.RandomCrop(width=224, height=224),
        A.Normalize(mean=[0.5, 0.5, 0.5],std=[0.225, 0.225, 0.225]),
        ToTensorV2()
    ])
    
    df_train, df_valid, df_test = make_fold(mode='train', fold=fold)
    train_dataset = SiimDataset(df_train, transform=train_transform)
    valid_dataset = SiimDataset(df_valid, transform=valid_transform)
    test_dataset = SiimDataset(df_test, transform=valid_transform)
    
    
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
        drop_last=True,
        # num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
    )
    
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = valid_batch_size,
        drop_last   = True,
        # num_workers = 4,
        pin_memory  = True
    )
    
    test_loader  = DataLoader(
        test_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = valid_batch_size,
        drop_last   = False,
        # num_workers = 4,
        pin_memory  = True
    )
        
    return train_loader, valid_loader, test_loader
    return train_loader, valid_loader, test_loader
    

def do_valid(model, valid_loader, epoch, batch, optimizer):
    model.eval()
    start_t = timer()
    correct, total_loss = 0, 0
    val_size = len(valid_loader.dataset)
    val_bsize = len(valid_loader)
    
    for t, (image, label) in enumerate(valid_loader):
        image, label = image.to(device), label.to(device, dtype=torch.float)
        # print(f'Valid | image: {image.shape}, label: {label.shape}')
        with torch.no_grad():
            logit = model(image)
            loss = F.cross_entropy(logit, label)
            # print(f'logit.shape: {logit.shape}')
            probability = F.softmax(logit,-1).cpu().data.numpy()
            probability = np.argmax(probability, axis=-1)
            label_copy = label.cpu().data.numpy()
            label_copy = np.argmax(label_copy, axis=-1)
            # print(f'prob: {probability}\nlabel: {label}')
            correct += sum(probability == label_copy)
            total_loss += loss
    valid_acc = correct/val_size
    valid_loss = total_loss/val_bsize
    log.write(f'{epoch}, {batch}, {valid_acc}, {valid_loss}\n')
    
    print(f'[Epoch {epoch}|{batch}] val_acc: {valid_acc:.3f} | val_loss: {valid_loss:.3f}')
    global best_acc
    if valid_acc > best_acc or best_acc == 0:
        best_acc = valid_acc
        if epoch % iter_log == 0 and epoch >= 5:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': epoch
            }, './checkpoint/%08d_model.pth' % (epoch))
    
def train(model, train_loader, valid_loader):
    rate, epoch = 0, 0
    
    optimizer = MADGRAD(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=start_lr, 
                        momentum= 0.9, 
                        weight_decay= 0, 
                        eps= 1e-06)
                        
    for iteration in range(num_iteration):
        for t, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device, dtype=torch.float)
            # print(f'train | image: {image.shape}, label: {label.shape}')
            model.train()
            optimizer.zero_grad()
            logit = model(image)
            loss = F.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            
            if t % iter_valid == 0 and t:
                do_valid(model=model, 
                        valid_loader=valid_loader, 
                        epoch=iteration, 
                        batch=t, 
                        optimizer=optimizer)



if __name__ == '__main__':
    train_loader, valid_loader, test_loader = create_siim_dataloader()

    # model = StudyNet()
    # model = model.to(device)
    # train(model, train_loader, valid_laoder)

    # meta_csv_path = './jpg_form/meta.csv'
    # jpg_df = pd.read_csv(meta_csv_path)
    # print(jpg_df)
    #
