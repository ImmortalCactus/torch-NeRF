import torch
from torch import nn
from torch.utils.data import DataLoader

import dataset
import render
import model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./nerf_synthetic/lego')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--freq_low', type=int, default=-2)
parser.add_argument('--freq_high', type=int, default=8)
parser.add_argument('--freq_dir_low', type=int, default=0)
parser.add_argument('--freq_dir_high', type=int, default=4)
parser.add_argument('--near', type=float, default=2)
parser.add_argument('--far', type=float, default=6)
parser.add_argument('--step_interval', type=int, default=8)
parser.add_argument('--clip', type=float, default=1)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'using {device} as device')


def train(train_dataset):
    encoder = model.IntegratedPositionalEncoder(freq_range=[-5, 5]).to(device)
    dir_encoder = model.PositionalEncoder(input_size=2, freq_range=[0, 4]).to(device)
    
    nerf_model = model.NerfModel(input_dim=encoder.output_size(), 
                                input_dim_dir=dir_encoder.output_size()).to(device)
    opt = torch.optim.Adam(nerf_model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    dataloader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True)

    nerf_model.train()
    step_counter = 0
    for epoch in range(args.epoch):
        for i, j, k in tqdm.tqdm(dataloader):
            o = i.to(device)
            d = j.to(device)
            ground_truth = k.to(device)
            t = torch.linspace(args.near, args.far, 65)
            t = t.to(device)
            t0 = t[:-1].reshape([1, -1, 1])
            t1 = t[1:].reshape([1, -1, 1])
            r = 1 / train_dataset.focal / 1.732 # sqrt(3)
            o = o.reshape([args.batch_size, 1, 3])
            d = d.reshape([args.batch_size, 1, 3])
            encoded = encoder(o, d, r, t0, t1)
            dirs = d / torch.norm(d, dim=-1, keepdim=True)
            sphericals = render.get_spherical(dirs)
            dirs_encoded = dir_encoder(sphericals)

            density, rgb = nerf_model(encoded, dirs_encoded)

            density = torch.unsqueeze(density, 2)
            delta = (t1-t0) * torch.norm(d, dim=-1, keepdim=True)
            result = model.integrate(density, rgb, delta)
            
            loss = loss_func(result, ground_truth)
            if torch.isnan(loss):
                print('STOP!')
                break
            loss.backward()
            nn.utils.clip_grad_norm_(nerf_model.parameters(), args.clip)
            opt.step()


if __name__ == "__main__":
    raw_data = dataset.load_synthetic(args.data_path, verbose=True)
    train_dataset = dataset.PixelDataset(raw_data['train'], verbose=True)

    train(train_dataset)
    