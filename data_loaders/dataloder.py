from torchvision import transforms
from torch.utils.data import DataLoader
from data_loaders.scfe import SCFE

import torch
import numpy as np

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def get_loader(config, mode="train"):
    # return the DataLoader
    #C://zhans//Desktop//dataset
    data_path = config.data_path
    #用Compose把多个步骤整合到一起
    transform = transforms.Compose([
    # transforms.Resize(config.img_size),# 已经是512*512的图像了
    transforms.ToTensor(),# 这里转换成C*H*W，range[0.0,1.0],需要是PIL读入 # normalize对于发现关系十分重要
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 对于mask不需要Normalize
    transform_mask = transforms.Compose([
        transforms.ToTensor()])

    if mode=="train":
        dataset_train = SCFE(data_path, transform=transform, mode= "train", transform_mask=transform_mask)
        dataset_test = SCFE(data_path, transform=transform, mode= "test", transform_mask=transform_mask)
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True,num_workers=2)

    if mode=="test":
        data_loader_train = None
        dataset_test = SCFE(data_path, transform=transform, mode= "test", transform_mask =transform_mask)

    data_loader_test = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             shuffle=False,num_workers=2)

    return data_loader_train