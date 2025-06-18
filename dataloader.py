# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:14:37 2023

@author: kylai

Keypoint: STD and MEAN are fixed in 0.5 due to the chip operation

"""

import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, PngImagePlugin
from pycocotools.coco import COCO
from timm.data import create_transform
from torch.utils.data import Dataset

from distribution import get_rank

# PIL.Image has the mechanism that restrict the image loading
# Set these to unlock the restriction
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def create_loader(
    data_root="/home/share/datasets/inhouse/images/MPS/v1.7/",
    mono=False,
    train='train',
    num_tasks=1, 
    rank=0, 
    batch_size=256,
    num_workers=4,
    pin_mem=True,
    flist=None,
    root_f=None,
    jlist=None,
    root_j=None,
    paste_root=['/home/share/kylai/ACERPROJ/acer_paste_0305_4class/'],
    auto_augment=False,
    rotation=False,
    train_size=[120, 160],
    val_size=[120, 160],
    dataset_type=['default']
):
    file_list = []
    class_list = []
    mask_list = None
    paste_class = None
    bg_list = None

    if flist and root_f:
        # import pdb;pdb.set_trace()
        f, c = _traverse_file_list(flist_path=flist, root=root_f)
        file_list += f
        class_list += c

    if data_root:
        f, c = _traverse_folder_image(data_root=data_root)
        file_list += f
        class_list += c

    if jlist and root_j:
        f, c = _traverse_json_file(json_path=jlist, root=root_j)
        file_list += f
        class_list += c

    if paste_root:
        mask_list, paste_class, bg_list = _traverse_folder_paste(paste_root)
    
    size = tuple(val_size) if train=='val' else tuple(train_size)
    img_transform = build_transform(size, train, auto_augment, rotation, mono)
    if get_rank() == 0:
        print("Total %d files in %s"%(len(file_list), train))
        print("Augment: ", img_transform)

    if isinstance(dataset_type, str):
        dataset_type = [dataset_type]
    dataset = [build_dataset(
        file_list=file_list,
        class_list=class_list,
        paste_mask=mask_list,
        paste_class=paste_class,
        paste_bg=bg_list,
        transform=img_transform,
        mono=mono,
        dataset_type=t
    ) for t in dataset_type]
    if len(dataset) > 1:
        dataset = torch.utils.data.ConcatDataset(dataset)
    else:
        dataset = dataset[0]
    
    if 'train' in train:
        sampler = torch.utils.data.DistributedSampler(
            dataset, 
            num_replicas=num_tasks, 
            rank=rank, 
            shuffle=True
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return data_loader

def build_dataset(
    file_list:list,
    class_list:list,
    paste_mask:list,
    paste_class:list,
    paste_bg:list,
    transform:transforms.Compose,
    mono:bool =False,
    size:tuple =(120, 160),
    dataset_type:str ='default'
):
    if dataset_type == 'default':
        dataset = default_dataset(
            file_list=file_list,
            class_list=class_list,
            transform=transform,
            mono=mono,
        )
    return dataset
class default_dataset(Dataset):
    ''' 
    Collection of openable path and its corresponding class w/ transform
    Args:
        file_list: Collection of openable image path
        class_list: Label of its corresponding file
        transforms: The operation before return image
        mono: The image is mono or not
    '''
    def __init__(
        self, 
        file_list:list,
        class_list:list,
        transform:transforms.Compose,
        mono:bool =False,
    ):
        self.mono = mono
        self.file_list = file_list
        self.class_idx = class_list
        self.img_transform = transform
        self.file_length = len(self.file_list)

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        if self.mono:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        img = self.img_transform(img)
        gt = self.class_idx[idx]
        return img, gt
    
def build_transform(size, train_type, auto_aug, rotation, mono):
    mean=(0.5,)
    std=(0.5,)
    if train_type == 'train':
        # this should always dispatch to transforms_imagenet_train
        if auto_aug:
            transform = create_transform(
                input_size=size,
                is_training=True,
                color_jitter=None,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                mean=mean,
                std=std,
            )
            del transform.transforms[2].ops[4]
            transform.transforms[2].ops[8] = gamma_correction()
            transform.transforms[0].scale = (0.8, 1.0)
            
            del transform.transforms[2].ops[3]
            if rotation:
                transform.transforms.insert(1, right_angle_rotation())
                del transform.transforms[3].ops[3]
            if mono:
                transform.transforms.insert(1, transforms.Grayscale())
            del transform.transforms[-1]
            del transform.transforms[2]
            del transform.transforms[2].ops[-1]
            del transform.transforms[2].ops[-1]
            del transform.transforms[2].ops[-1]
            del transform.transforms[2].ops[-1]
            del transform.transforms[2].ops[-2]
            
        else:
            # TODO Write a config-like flow to control the augmentation 
            transform_list = [
                transforms.Grayscale(),
                transforms.RandomOrder([
                    transforms.ColorJitter(
                        brightness=(0.9, 1.2),
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1
                    ),
                    transforms.RandomEqualize(),
                    transforms.RandomAutocontrast(),
                    # transforms.RandomPosterize(bits=5),
                    # cut_out(),
                ]),
                
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
            if not mono:
                del transform_list[0]
            transform = transforms.Compose(transform_list)
        return transform
    
    elif train_type == 'pretrain':
        transform_list = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        if not mono:
            del transform_list[2]
        return transforms.Compose(transform_list)
    
    # val
    transform_list = [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ]
    # val
    transform_list = [
        preprocess_image(),
    ]
    if not mono:
        del transform_list[0]
    return transforms.Compose(transform_list)

def preprocess_image():
    def _preprocess_image(img):
        H=160
        W=120
        img = np.array(img, dtype=np.uint8)
        h, w = img.shape
        r = min(H/h, W/w)
        rh = int(h*r)
        rw = int(w*r)
        # img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        # img = cv2.copyMakeBorder(img, (H+1-rh)//2, (H-rh)//2, (W+1-rw)//2, (W-rw)//2, cv2.BORDER_CONSTANT, 0) # TBLR to 320,256
        img = torch.from_numpy(img)
        img = img.float() / 256
        return img.unsqueeze(0)
    return _preprocess_image

class right_angle_rotation:
    def __init__(self, angle=None):
        self.angle = angle
    def __call__(self, x):
        if self.angle:
            return transforms.functional.rotate(x, self.angle)
        angle = int(np.random.choice([0, 90, 180, 270], 1)[0])
        return transforms.functional.rotate(x, angle)

class gamma_correction:
    """
    An augmentation that support gamma correction and linear light adjustment
    
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            gamma = np.random.uniform(0.5, 1.5)
            x = transforms.functional.adjust_gamma(x, gamma)

        return x

def cut_out(mask_size=20, p=0.5):
    """ Cut our a random area in an image """
    def _cutout(image):
        if np.random.random() > p:
            return image
        
        mode = image.mode
        image = np.array(image, dtype=np.uint8)
        h, w = image.shape

        xmin = np.random.randint(0, w)
        ymin = np.random.randint(0, h)
        xmax = xmin + np.random.randint(0, mask_size)
        ymax = ymin + np.random.randint(0, mask_size)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        image[ymin:ymax, xmin:xmax] = 0
        image = Image.fromarray(image, mode)
        return image

    return _cutout

def _traverse_folder_image(data_root):
    ext = ['PNG', 'JPEG', 'jpg', 'png']
    # The highest default priority mapping
    # Make sure it will overwrite the traverse result
    convert = {
        '75_angle': 0,
        '45_angle': 1,
        'leave': 2,
        'multi_person': 3,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5
    }
    classes = set()
    file_list = []
    cls_list = []
    
    for p in data_root if isinstance(data_root, list) else [data_root]:
        cls = sorted(entry.name for entry in os.scandir(p) if entry.is_dir())
        classes = classes.union(set(cls))
        path = Path(p)
        flist = []
        clslist = []
        for target_class in cls:
            f = []
            for e in ext:
                f += glob.glob(str(path / target_class / '**' / ('*.'+e)), recursive=True)
            flist += f
            clslist += [target_class for _ in f]
            
        file_list += flist
        cls_list += clslist
        if get_rank() == 0:
            print('%d files in %s'%(len(flist), str(path / '**' / ('*.'))), ext)
    
    # Class index are determined when all class are collected
    # Update after class_to_idx produced to make sure the default setting
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
    class_to_idx.update(convert)
    for i, c in enumerate(cls_list):
        cls_list[i] = class_to_idx[c]
    assert len(file_list) == len(cls_list)
    return file_list, cls_list

def _traverse_folder_paste(data_root):
    ext = ['PNG', 'JPEG', 'jpg', 'png']
    # The highest default priority mapping
    # Make sure it will overwrite the traverse result
    convert = {
        '75_angle': 0,
        '45_angle': 1,
        'leave': 2,
        'multi_person': 3,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5
    }
    classes = set()
    mask_list = []
    cls_list = []
    bg_list = []
    
    for p in data_root if isinstance(data_root, list) else [data_root]:
        p = os.path.join(p, 'background')
        path = Path(p)
        bg = []
        for e in ext:
            bg += glob.glob(str(path / '**' / ('*.'+e)), recursive=True)
        bg_list += bg
        if get_rank() == 0:
            print('%d bgs in %s'%(len(bg), str(path / '**' / ('*.'))), ext)

    for p in data_root if isinstance(data_root, list) else [data_root]:
        p = os.path.join(p, 'mask')
        cls = sorted(entry.name for entry in os.scandir(p) if entry.is_dir())
        classes = classes.union(set(cls))
        path = Path(p)
        mlist = []
        clslist = []
        for target_class in cls:
            f = []
            for e in ext:
                f += glob.glob(str(path / target_class / '**' / ('*.'+e)), recursive=True)
            mlist += f
            clslist += [target_class for _ in f]
            if get_rank() == 0:
                print('%3d class %s masks in %s'%(len(f), target_class, str(path / target_class / '**' / ('*.'))), ext)
            
        mask_list += mlist
        cls_list += clslist
    
    # Class index are determined when all class are collected
    # Update after class_to_idx produced to make sure the default setting
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
    class_to_idx.update(convert)
    for i, c in enumerate(cls_list):
        cls_list[i] = class_to_idx[c]
    assert len(mask_list) == len(cls_list)
    return mask_list, cls_list, bg_list

def _traverse_file_list(flist_path, root):
    file_list = []
    cls_list = []
    assert len(flist_path) == len(root) # hint: string or list
    # import pdb; pdb.set_trace() 
    for fpath, froot in zip(flist_path, root):
        fpath = Path(fpath)
        fpath = glob.glob(str(fpath / '**' / '*.txt'), recursive=True)
        for p in fpath:
            txt_name = p.split(os.sep)[-1].split('.')[0]
            if txt_name.isnumeric():
                idx = int(txt_name)
                with open(p, 'r') as f:
                    flist = f.read().splitlines()
                # flist = [os.path.join(froot, txt_name, x+'.PNG') for x in flist]
                flist = [os.path.join(froot, txt_name, x+'.JPEG') for x in flist]
                clslist = [idx for _ in range(len(flist))]
                if get_rank() == 0:
                    print('%d path in %s'%(len(flist), p))
                file_list += flist
                cls_list += clslist

    assert len(file_list) == len(cls_list)
    return file_list, cls_list

def _traverse_json_file(json_path, root):
    file_list = []
    cls_list = []
    num = 0
    assert len(json_path) == len(root)
    for jpath, jroot in zip(json_path, root):
        with open(jpath, 'r') as j:
            ann = json.load(j)
        
        file_list += [os.path.join(jroot, x['file_name']) for x in ann['images']]
        vww = COCO(jpath)
        ids = list([x['id'] for x in ann['images']])
        cls_list = [vww.loadAnns(vww.getAnnIds(x)) for x in ids]
        cls_list = [x[0]['category_id'] if x != [] else 0 for x in cls_list ]
        print('%d files in %s'%((len(cls_list)), jpath))
        
    assert len(file_list) == len(cls_list)
    return file_list, cls_list

def appendix(num):
    # Top 16 classes from the result of mcunet-in2  
    classes_ = [
        'n13037406',
        'n01622779',
        'n03733131',
        'n11939491',
        'n02391049',
        'n02006656',
        'n03662601',
        'n03393912',
        'n02116738',
        'n01534433',
        'n02025239',
        'n03841143',
        'n02489166',
        'n12985857',
        'n12057211',
        'n13044778' 
    ][:num]
    # Worst 16 classes from the result of mcunet-in4
    classes_ = [
        'n04152593',
        'n04525038',
        'n04355933',
        'n03372029',
        'n03532672',
        'n04270147',
        'n03045698',
        'n03633091',
        'n04154565',
        'n03866082',
        'n03832673',
        'n04286575',
        'n02123159',
        'n04560804',
        'n03041632',
        'n02669723'
    ][:num]
    # top hard dog
    classes_ = [
        'n02111889',
        'n02111129',
        'n02110341',
        'n02110958',
        'n02109525',
        'n02108551',
        'n02110806',
        'n02108422',
        'n02108915',
        'n02110627',
        'n02110063',
        'n02111277',
        'n02111500',
        'n02109961',
        'n02109047',
        'n02110185'
    ][:num]