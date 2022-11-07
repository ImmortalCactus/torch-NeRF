import os
import imageio.v3 as imageio
import numpy as np
import json
import typing

from torch.utils.data import Dataset

import render
"""
- 'train'
    - 'images'
    - matrices
    - height
    - width
    - focal
- 'test'
- 'val'
"""

def load_synthetic(basedir: str='./nerf_synthetic/lego', verbose=False):
    ret = {'train': {}, 'test': {}, 'val': {}}
    if verbose: print(f'loading from {basedir}.')
    for d in ret:
        if verbose: print(f'loading {d} dataset...')
        with open(os.path.join(basedir, f'transforms_{d}.json')) as f:
            meta_data = json.load(f)
        dataset = ret[d]
        dataset['images'] = []
        dataset['matrices'] = []
        dataset['height'] = None
        dataset['width'] = None
        dataset['focal'] = None
        for frame in meta_data['frames']:
            img = imageio.imread(os.path.join(basedir, frame['file_path'] + '.png'))
            dataset['images'].append(img)

            if dataset['height'] is None:
                dataset['height'], dataset['width'] = img.shape[:2]
                dataset['focal'] = dataset['width']/np.arctan(meta_data['camera_angle_x']/2)

            matrix = np.array(frame['transform_matrix'])
            dataset['matrices'].append(matrix)
    if verbose: print('done loading.')
    return ret

class PixelDataset(Dataset):
    def __init__(self, data_dict, split='train', verbose=False):
        self.split = split
        if verbose: print('creating dataset')
        self.images = np.stack(data_dict['images'])
        self.images = self.images.astype(np.float32) / 255
        if verbose: print('- images')
        self.matrices = np.stack(data_dict['matrices'])
        if verbose: print('- matrices')
        self.height = data_dict['height']
        self.width = data_dict['width']
        self.focal = data_dict['focal']
        
        self.rays = {}
        self.rays['origins'], self.rays['directions'] = render.get_rays(
            self.height, 
            self.width, 
            self.focal, 
            self.matrices)
        if verbose: print('- rays')

    def __len__(self):
        return self.height * self.width * self.images.shape[0]

    def __getitem__(self, idx):
        n_img = idx // (self.height * self.width)
        i = idx % (self.height * self.width) // self.width
        j = idx % self.width
        return self.rays['origins'][n_img], self.rays['directions'][n_img, i, j], self.images[n_img, i, j]
