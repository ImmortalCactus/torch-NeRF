import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import dataset
import render
import model
import argparse
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
parser.add_argument('--fine_sample_per_ray', type=int, default=135)
parser.add_argument('--step_interval', type=int, default=8)
parser.add_argument('--clip', type=float, default=1)
parser.add_argument('--ckpt', type=str, default='ckpt/model.ckpt')
parser.add_argument('--from_ckpt', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--lr_low', type=float, default=5e-6)
parser.add_argument('--lambda_loss', type=float, default=0.1)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')
print(f'using {device} as device')

def forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder):
    encoded = encoder(o, d, r, t_vals[:, :-1, :], t_vals[:, 1:, :])
    dirs = d / torch.norm(d, dim=-1, keepdim=True)
    sphericals = render.get_spherical(dirs)
    dirs_encoded = dir_encoder(sphericals)
    density, rgb = nerf_model(encoded, dirs_encoded)
    density = torch.unsqueeze(density, 2)
    delta_t = t_vals[:, 1:, :] - t_vals[:, :-1, :]
    delta_t *= torch.norm(d, dim=-1, keepdim=True)
    weights = render.integrate_weights(density, delta_t)
    rgba = torch.cat([rgb, torch.ones(*weights.shape).to(device)], dim=2)
    result = torch.sum(rgba * weights, dim=1)
    return weights, result

def train(train_dataset, nerf_model, encoder, dir_encoder):
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

    if args.from_ckpt:
        checkpoint = torch.load(args.ckpt)
        nerf_model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        sch.load_state_dict(checkpoint['sch'])
    
    loss_func = nn.MSELoss()

    dataloader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True)

    nerf_model.train()
    step_counter = 0
    count = 0
    for epoch in range(args.epoch):
        for it, (o, d, ground_truth) in enumerate(pbar:= tqdm.tqdm(dataloader)):
            step_counter += 1
            o = o.to(device)
            d = d.to(device)
            ground_truth = ground_truth.to(device)
            r = 1 / train_dataset.focal / 1.732 # sqrt(3)
            o = o.reshape([args.batch_size, 1, 3])
            d = d.reshape([args.batch_size, 1, 3])

            t_vals = render.gen_intervals(args.near, args.far, args.batch_size, args.sample_per_ray)
            t_vals = t_vals.to(device)

            weights_coarse, result_coarse = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

            fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
            fine_t_vals = fine_sampler.sample(args.fine_sample_per_ray + 1, random=True).detach()

            weights_fine, result_fine = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder)

            coarse_loss = loss_func(result_coarse, ground_truth)
            fine_loss = loss_func(result_fine, ground_truth)
            loss = fine_loss + args.lambda_loss * coarse_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            summary = [
                f'epoch {epoch}',
                f'iteration {it}/{len(dataloader)}',
                f'loss: {coarse_loss:.3e}/{fine_loss:.3e}/{loss:.3e}',
                f'alpha: {torch.max(result_fine[..., -1:]):.2f}/{torch.min(result_fine[..., -1:]):.2f}'
            ]
            pbar.set_description(' | '.join(summary))
        torch.save({
            'model': nerf_model.state_dict(),
            'opt': opt.state_dict(),
            'sch': sch.state_dict()
            }, args.ckpt)

def test(test_dataset, nerf_model, encoder, dir_encoder):
    dataloader = DataLoader(test_dataset, 
                            batch_size=test_dataset.width,
                            shuffle=False)
    nerf_model.load_state_dict(torch.load(args.ckpt['model']))
    nerf_model.eval()
    with torch.no_grad():
        rows = []
        idx_png = 0
        for o, d, ground_truth in tqdm.tqdm(dataloader):
            o = o.to(device)
            d = d.to(device)
            ground_truth = ground_truth.to(device)
            r = 1 / test_dataset.focal / 1.732 # sqrt(3)
            o = o.reshape([test_dataset.width, 1, 3])
            d = d.reshape([test_dataset.width, 1, 3])

            t_vals = render.gen_intervals(args.near, args.far, test_dataset.width, args.sample_per_ray)
            t_vals = t_vals.to(device)

            weights_coarse, _ = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

            fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
            fine_t_vals = fine_sampler.sample(args.fine_sample_per_ray + 1, random=True).detach()

            _, result_fine = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder)

            rows.append(result_fine)
            if len(rows) % test_dataset.height == 0:
                img = torch.stack(rows)
                img = img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGBA')
                rows = []
                img.save(f'output/{idx_png}.png')
                idx_png += 1


if __name__ == "__main__":
    encoder = model.IntegratedPositionalEncoder(freq_range=[args.freq_low, args.freq_high]).to(device)
    dir_encoder = model.PositionalEncoder(input_size=2, freq_range=[0, 4]).to(device)
    nerf_model = model.NerfModel(input_dim=encoder.output_size(), 
                                input_dim_dir=dir_encoder.output_size()).to(device)
    raw_data = dataset.load_synthetic(args.data_path, verbose=True)
    
    if args.mode == 'train':
        train_dataset = dataset.PixelDataset(raw_data['train'], verbose=True)
        train(train_dataset, nerf_model, encoder, dir_encoder)
    elif args.mode == 'test':
        test_dataset = dataset.PixelDataset(raw_data['test'], verbose=True)
        test(test_dataset, nerf_model, encoder, dir_encoder)
