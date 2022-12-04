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
                dataset['focal'] = dataset['width'] / 2. /np.arctan(meta_data['camera_angle_x']/2)

            matrix = np.array(frame['transform_matrix'])
            dataset['matrices'].append(matrix)
    if verbose: print('done loading.')
    return ret

class PixelDataset(Dataset):
    def __init__(self, data_dict, split='train', mip_level=1):
        self.mip_level = mip_level
        self.split = split
        self.matrices = np.stack(data_dict['matrices'])

        self.images = []
        self.heights = []
        self.widths = []
        self.focals = []
        self.rays = []
        self.counts = []
        self.len = 0

        def half(x):
            x = (x[:, ::2, ...] + x[:, 1::2, ...]) / 2
            x = (x[:, :, ::2, ...] + x[:, :, 1::2, ...]) / 2
            return x

        for i in range(mip_level):
            if i == 0:
                imgs = np.stack(data_dict['images'])
                imgs = imgs / 255.
                self.images.append(imgs)
                
                self.heights.append(data_dict['height'])
                self.widths.append(data_dict['width'])
                self.focals.append(data_dict['focal'])
                o, d = render.get_rays(
                    self.heights[0], 
                    self.widths[0],
                    self.focals[0], 
                    self.matrices)
                self.rays.append({'origins': o, 'directions': d})
            else:
                self.images.append(half(self.images[i-1]))
                self.heights.append(self.heights[i-1]//2)
                self.widths.append(self.widths[i-1]//2)
                self.focals.append(self.focals[i-1]/2)
                o = self.rays[i-1]['origins']
                d = self.rays[i-1]['directions']
                self.rays.append({'origins': o, 'directions': half(d)})
            
            
            self.counts.append(self.images[i].shape[0] * self.heights[i] * self.widths[i])
            self.len += self.counts[i]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n_level = 0
        while idx >= self.counts[n_level]:
            idx -= self.counts[n_level]
            n_level += 1
        n_img = idx // (self.heights[n_level] * self.widths[n_level])
        idx -= n_img * (self.heights[n_level] * self.widths[n_level])
        i = idx // self.widths[n_level]
        j = idx % self.widths[n_level]
        return self.rays[n_level]['origins'][n_img], self.rays[n_level]['directions'][n_img, i, j], self.images[n_level][n_img, i, j], 1/self.focals[n_level], (2.**n_level)
