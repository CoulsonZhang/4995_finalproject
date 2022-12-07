
import os, pydicom
import pandas as pd 
import numpy as np

from file import *
from PIL import Image
from tqdm.auto import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im

def convert_to_jpg(study_level_csv='../siim-covid19-detection/train_study_level.csv', meta_csv_p='./meta.csv', 
                    save_dir='./jpg_form', dataset_p='../siim-covid19-detection',
                    img_size=512):
    train_csv = pd.read_csv(study_level_csv)
    image_id = []
    origin_id = []
    dim0 = []
    dim1 = []
    splits = []
    labels = []
    

    for split in ['test', 'train']:
        img_save_dir = save_dir + f'/{split}/'
        os.makedirs(img_save_dir, exist_ok=True)
    
        data_path = dataset_p + f'/{split}'
        for root, dirs, filenames in tqdm(os.walk(data_path)):
            for file in filenames:
                # set keep_ratio=True to have original aspect ratio
                #print(f'{root}')
                csv_id = root.split('/')[-2] + '_study'
                #print(csv_id)
                id_row = train_csv.loc[train_csv['id'] == csv_id]
                #print(id_row)
                id_row = id_row.loc[:, id_row.columns != 'id']
                #print(id_row)
                label = id_row.values.tolist()
                
                xray = read_xray(os.path.join(root, file))
                im = resize(xray, size=img_size)  
                img_save_path = os.path.join(img_save_dir, file.replace('dcm', 'jpg'))
                im.save(img_save_path)
                #print(img_save_path)
                
                single_img_id = file.replace('.dcm', '')
                image_id.append(single_img_id)
                origin_id.append(csv_id)
                dim0.append(xray.shape[0])
                dim1.append(xray.shape[1])
                labels.append(label)
                splits.append(split)
                
                #print(f'id: {single_img_id}, origin: {csv_id}, label: {label}, split: {split}')
                
    df = pd.DataFrame.from_dict({'id': image_id, 'origin_train_id': origin_id, 'dim0': dim0, 'dim1': dim1, 'label': labels, 'split': splits})
    df.to_csv(meta_csv_p, index=False)
    return meta_csv_p
    
    
    
    
def make_fold(mode='train', fold=4, data_dir='./jpg_form'):
    if 'train' in mode:
        df_study = pd.read_csv(data_dir+'/meta.csv')
        df_fold  = pd.read_csv(data_dir+'/df_fold_rand830.csv')
        df_meta  = pd.read_csv(data_dir+'/df_meta.csv')
    
        df_study.loc[:, 'origin_train_id'] = df_study.origin_train_id.str.replace('_study', '')
        df_study = df_study.rename(columns={'origin_train_id': 'study_id'})
        
        #---
        df = df_study.copy()
        df = df.merge(df_fold, on='study_id')
        # df = df.merge(df_meta, left_on='study_id', right_on='study')
    
        duplicate = read_list_from_file(data_dir + '/duplicate.txt')
        df = df[~df['id'].isin(duplicate)]
    
        #---
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_rest = df[df.fold == fold].reset_index(drop=True)
        middle = df_rest.size // 2
        df_valid = df_rest.iloc[:middle, :]
        df_test = df_rest.iloc[middle:, :]
        return df_train, df_valid, df_test

    if 'test' in mode:
        df_meta  = pd.read_csv(data_dir+'/meta.csv')
        df_valid = df_meta[df_meta['split']=='test'].copy()
    
        df_valid = df_valid.assign(label=[0,0,0,0])
        df_valid = df_valid.reset_index(drop=True)
        return df_valid



