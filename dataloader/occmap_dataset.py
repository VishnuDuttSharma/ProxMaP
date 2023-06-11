__author__ = "Vishnu Dutt Sharma"

import pandas as pd
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

class OccMapDataset(data.Dataset):
    def __init__(self, filename='./updated_description_ang0.csv', 
                 transform=None, 
                 input_dir='./data/inp_data/', 
                 target_dir='./data/gt_data/', 
                 mode='train', 
                 odds_to_prob=True, 
                 prob_scale=10., 
                 count_to_odd=False, 
                 to_class=False,
                 with_seg=False,
                 seg_dir='./data/inp_seg_data/'):
        
        # Your code
        df = pd.read_csv(filename)
        df = df[df['free_perc'] <= 80]

        df['FloorID'] = df['FloorName'].apply(lambda x: int(x[-3:]))
        if mode == 'train':
            df = df[df['FloorID'] <= 220]
        else:
            df = df[df['FloorID'] > 220]

        self.filepaths = df['Filename'].values

        self.transform = transform
        self.input_dir = input_dir
        self.target_dir = target_dir

        self.odds_to_prob = odds_to_prob
        self.prob_scale = prob_scale

        self.mode = mode
        self.count_to_odd = count_to_odd
        self.to_class = to_class

        self.with_seg = with_seg
        self.seg_dir = seg_dir

    def __len__(self):
        # Your code
        return len(self.filepaths)

    def __getitem__(self, index):
        # Your code
        filename = self.filepaths[index]
        inp_img = np.load(f'{self.input_dir}/{self.filepaths[index]}.npy')
        tgt_img = np.load(f'{self.target_dir}/{self.filepaths[index]}.npy')

        if self.count_to_odd:
            inp_img = inp_img.astype(np.float) * 0.01
            tgt_img = tgt_img.astype(np.float) * 0.01

        data_dict = {'input image': inp_img[:, :], 'target image': tgt_img[ :, :]}
        data_dict['input image'] *= self.prob_scale
        data_dict['target image'] *= self.prob_scale

        if self.odds_to_prob:
            o2p_func = lambda x: np.exp(x)/(1. + np.exp(x))
            data_dict['input image'] = o2p_func(data_dict['input image'])
            data_dict['target image'] = o2p_func(data_dict['target image'])
        
        if self.to_class:
            inp_class = np.zeros(data_dict['input image'].shape + (3,))
            inp_class[data_dict['input image'] < 0.495, 0] = 1
            inp_class[(data_dict['input image'] >= 0.495) & (data_dict['input image'] <= 0.505), 1] = 1
            inp_class[data_dict['input image'] > 0.505, 2] = 1
            data_dict['input image'] = inp_class.astype(np.float32)

            out_class = np.zeros(data_dict['target image'].shape)
            out_class[data_dict['target image'] < 0.495] = 0
            out_class[(data_dict['target image'] >= 0.495) & (data_dict['target image'] <= 0.505)] = 1
            out_class[data_dict['target image'] > 0.505] = 2
            data_dict['target image'] = out_class.astype(int)

        if self.with_seg:
            seg_img = np.load(f'{self.seg_dir}/{self.filepaths[index]}.npy')
            seg_img = (seg_img > 0).astype(np.float32)
            data_dict['input image'] = np.concatenate([data_dict['input image'], seg_img.transpose(1,2,0)], axis=2)

        
        if self.transform is not None:
            data_dict['input image'] = self.transform(data_dict['input image'])
            if self.to_class:
                data_dict['target image'] = torch.from_numpy(data_dict['target image'][np.newaxis,...]).long()
            else:
                data_dict['target image'] = self.transform(data_dict['target image'])

        if self.mode == 'train':
            if np.random.random() > 0.5:
                data_dict['input image'] = TF.vflip(data_dict['input image'])
                data_dict['target image'] = TF.vflip(data_dict['target image'])
            '''
            if np.random.random() > 0.5:
                data_dict['input image'] = TF.hflip(data_dict['input image'])
                data_dict['target image'] = TF.hflip(data_dict['target image'])
            '''


        return data_dict

class OccMap_360_Dataset(data.Dataset):
    def __init__(self, filename='./description_ang0.csv', transform=None, input_dir='./inp_data/', target_dir='./fullview_data/',
                        mode='train', odds_to_prob=True, scale=10., count_to_odd=False, to_class=False):
        # Your code
        df = pd.read_csv(filename)
        df = df[df['free_perc'] <= 80]

        df['FloorID'] = df['FloorName'].apply(lambda x: int(x[-3:]))
        if mode == 'train':
            df = df[df['FloorID'] <= 220]
        else:
            df = df[df['FloorID'] > 220]

        self.filepaths = df['Filename'].values

        self.transform = transform
        self.input_dir = input_dir
        self.target_dir = target_dir

        self.odds_to_prob = odds_to_prob
        self.scale = scale

        self.mode = mode
        self.count_to_odd = count_to_odd
        self.to_class = to_class


    def __len__(self):
        # Your code
        return len(self.filepaths)

    def __getitem__(self, index):
        # Your code
        filename = self.filepaths[index]
        orig_inp_img = np.load(f'{self.input_dir}/{self.filepaths[index]}.npy')
        inp_img = 0*orig_inp_img.copy()
        inp_img[:,128:] = orig_inp_img[:,:128]

        tgt_img = np.load(f'{self.target_dir}/{self.filepaths[index]}.npy')

        if self.count_to_odd:
            inp_img = inp_img.astype(np.float) * 0.01
            tgt_img = tgt_img.astype(np.float) * 0.01

        data_dict = {'input image': inp_img[:, :], 'target image': tgt_img[ :, :]}
        data_dict['input image'] *= self.scale
        data_dict['target image'] *= self.scale

        if self.odds_to_prob:
            o2p_func = lambda x: np.exp(x)/(1. + np.exp(x))
            data_dict['input image'] = o2p_func(data_dict['input image'])
            data_dict['target image'] = o2p_func(data_dict['target image'])

        if self.to_class:
            inp_class = np.zeros(data_dict['input image'].shape + (3,))
            inp_class[data_dict['input image'] < 0.495, 0] = 1
            inp_class[(data_dict['input image'] >= 0.495) & (data_dict['input image'] <= 0.505), 1] = 1
            inp_class[data_dict['input image'] > 0.505, 2] = 1
            data_dict['input image'] = inp_class.astype(np.float32)

            out_class = np.zeros(data_dict['target image'].shape)
            out_class[data_dict['target image'] < 0.495] = 0
            out_class[(data_dict['target image'] >= 0.495) & (data_dict['target image'] <= 0.505)] = 1
            out_class[data_dict['target image'] > 0.505] = 2
            data_dict['target image'] = out_class.astype(int)

        if self.transform is not None:
            data_dict['input image'] = self.transform(data_dict['input image'])
            if self.to_class:
                data_dict['target image'] = torch.from_numpy(data_dict['target image'][np.newaxis,...]).long()
            else:
                data_dict['target image'] = self.transform(data_dict['target image'])

        if self.mode == 'train':
            if np.random.random() > 0.5:
                data_dict['input image'] = TF.vflip(data_dict['input image'])
                data_dict['target image'] = TF.vflip(data_dict['target image'])
            '''
            if np.random.random() > 0.5:
                data_dict['input image'] = TF.hflip(data_dict['input image'])
                data_dict['target image'] = TF.hflip(data_dict['target image'])
            '''


        return data_dict
