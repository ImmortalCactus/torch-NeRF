import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import dataset
import render
import model
import argparse
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='./nerf_synthetic/lego')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--freq_low', type=int, default=-2)
parser.add_argument('--freq_high', type=int, default=12)
parser.add_argument('--freq_dir_low', type=int, default=0)
parser.add_argument('--freq_dir_high', type=int, default=4)
parser.add_argument('--near', type=float, default=2)
parser.add_argument('--far', type=float, default=6)
parser.add_argument('--sample_per_ray', type=int, default=128)
parser.add_argument('--step_interval', type=int, default=8)
parser.add_argument('--clip', type=float, default=1)
parser.add_argument('--ckpt', type=str, default='ckpt/model.ckpt')
parser.add_argument('--from_ckpt', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--lr_low', type=float, default=5e-6)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')
print(f'using {device} as device')


def train(train_dataset):
    encoder = model.IntegratedPositionalEncoder(freq_range=[args.freq_low, args.freq_high]).to(device)
    # encoder = model.PositionalEncoder(input_size=3, freq_range=[args.freq_low, args.freq_high]).to(device)
    dir_encoder = model.PositionalEncoder(input_size=2, freq_range=[0, 4]).to(device)
    
    nerf_model = model.NerfModel(input_dim=encoder.output_size(), 
                                input_dim_dir=dir_encoder.output_size()).to(device)
    if args.from_ckpt:
        nerf_model.load_state_dict(torch.load(args.ckpt))
    opt = torch.optim.Adam(nerf_model.parameters(), lr=args.lr)

    def lr_lambda(i):
        if i < args.warmup:
            ret = args.lr_low + (args.lr - args.lr_low) * np.sin(np.pi / 2 * i / args.warmup)
        else:
            ret = 1
            num_iter = args.epoch * len(train_dataset) // args.batch_size
            w = i / num_iter
            ret *= np.exp((1-w) * np.log(args.lr) + w * np.log(args.lr_low))
        return ret / args.lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_func = nn.MSELoss()

    dataloader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True)

    nerf_model.train()
    step_counter = 0
    count = 0
    for epoch in range(args.epoch):
        for i, j, k in tqdm.tqdm(dataloader):
            step_counter += 1
            o = i.to(device)
            d = j.to(device)
            ground_truth = k.to(device)
            t = torch.linspace(args.near, args.far, args.sample_per_ray)
            t = t.to(device)
            t0 = t[:-1].reshape([1, -1, 1])
            t1 = t[1:].reshape([1, -1, 1])
            random_t = torch.rand(o.shape[0], t0.shape[2], 1).to(device)
            t_vals = (t0 * random_t + t1 * (1 - random_t)) # 128 - 1
            onez = torch.ones(t_vals.shape[0], 1, t_vals.shape[2]).to(device)
            t_vals = torch.cat([onez * args.near, t_vals, onez * args.far] , dim=1)
            r = 1 / train_dataset.focal / 1.732 # sqrt(3)
            o = o.reshape([args.batch_size, 1, 3])
            d = d.reshape([args.batch_size, 1, 3])
            encoded = encoder(o, d, r, t_vals[:, :-1, :], t_vals[:, 1:, :])
            """
            random_t = torch.rand(o.shape[0], t0.shape[2], 1).to(device)
            t_vals = (t0 * random_t + t1 * (1 - random_t))
            encoded = encoder(o + d * t_vals)
            """
            dirs = d / torch.norm(d, dim=-1, keepdim=True)
            sphericals = render.get_spherical(dirs)
            dirs_encoded = dir_encoder(sphericals)

            density, rgb = nerf_model(encoded, dirs_encoded)

            density = torch.unsqueeze(density, 2)
            delta = (t_vals[:, 1:, :] - t_vals[:, :-1, :]) * torch.norm(d, dim=-1, keepdim=True)
            result = model.integrate(density, rgb, delta)
            loss = loss_func(result, ground_truth)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            count += 1
            if count == 100:
                count = 0
                print(f'lr: {opt.param_groups[0]["lr"]}')
                print(f'loss: {loss}')
                print(f'max_alpha: {torch.max(result[..., -1])}')
                print(f'min_alpha: {torch.min(result[..., -1])}')
            if step_counter == 5000:
                torch.save(nerf_model.state_dict(), args.ckpt)
        torch.save(nerf_model.state_dict(), args.ckpt)

def test(test_dataset):
    dataloader = DataLoader(test_dataset, 
                            batch_size=800,
                            shuffle=False)
    encoder = model.IntegratedPositionalEncoder(freq_range=[args.freq_low, args.freq_high]).to(device)
    dir_encoder = model.PositionalEncoder(input_size=2, freq_range=[0, 4]).to(device)
    nerf_model = model.NerfModel(input_dim=encoder.output_size(), 
                                input_dim_dir=dir_encoder.output_size()).to(device)
    nerf_model.load_state_dict(torch.load(args.ckpt))
    nerf_model.eval()
    with torch.no_grad():
        rows = []
        row_idx = 0
        idx_png = 0
        for i, j, k in tqdm.tqdm(dataloader):
            row_idx += 1
            o = i.to(device)
            d = j.to(device)
            ground_truth = k.to(device)
            t = torch.linspace(args.near, args.far, args.sample_per_ray)
            t = t.to(device)
            t0 = t[:-1].reshape([1, -1, 1])
            t1 = t[1:].reshape([1, -1, 1])
            random_t = torch.rand(o.shape[0], t0.shape[2], 1).to(device)
            t_vals = (t0 * random_t + t1 * (1 - random_t)) # 128 - 1
            onez = torch.ones(t_vals.shape[0], 1, t_vals.shape[2]).to(device)
            t_vals = torch.cat([onez * args.near, t_vals, onez * args.far] , dim=1)
            r = 1 / test_dataset.focal / 1.732 # sqrt(3)
            o = o.reshape([800, 1, 3])
            d = d.reshape([800, 1, 3])
            encoded = encoder(o, d, r, t_vals[:, :-1, :], t_vals[:, 1:, :])
            dirs = d / torch.norm(d, dim=-1, keepdim=True)
            sphericals = render.get_spherical(dirs)
            dirs_encoded = dir_encoder(sphericals)

            density, rgb = nerf_model(encoded, dirs_encoded)

            density = torch.unsqueeze(density, 2)
            delta = (t_vals[:, 1:, :] - t_vals[:, :-1, :]) * torch.norm(d, dim=-1, keepdim=True)
            result = model.integrate(density, rgb, delta)
            rows.append(result)
            if row_idx % 800 == 0:
                img = torch.stack(rows)
                img = img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGBA')
                rows = []
                img.save(f'output/{idx_png}.png')
                idx_png += 1


if __name__ == "__main__":
    raw_data = dataset.load_synthetic(args.data_path, verbose=True)
    
    if args.mode == 'train':
        train_dataset = dataset.PixelDataset(raw_data['train'], verbose=True)
        train(train_dataset)
    elif args.mode == 'test':
        test_dataset = dataset.PixelDataset(raw_data['test'], verbose=True)
        test(test_dataset)
    elif args.mode == 'debug':
        t = torch.linspace(-4, 4, 1025)
        t0 = t[:-1].reshape([1, -1, 1])
        t1 = t[1:].reshape([1, -1, 1])
