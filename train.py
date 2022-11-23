import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import tqdm
import dataset
import render
import model
import argparse
import yaml
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

if torch.cuda.is_available():
    device = torch.device(config['resource']['device'])
else:
    device = torch.device('cpu')
print(f'using {device} as device')

def forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder):
    encoded = encoder(o, d, r, t_vals[:, :-1, :], t_vals[:, 1:, :])
    dirs = d / torch.norm(d, dim=-1, keepdim=True)
    sphericals = render.get_spherical(dirs)
    dirs_encoded = dir_encoder(sphericals)
    density, rgb = nerf_model(encoded, dirs_encoded)
    delta_t = t_vals[:, 1:, :] - t_vals[:, :-1, :]
    delta_t *= torch.norm(d, dim=-1, keepdim=True)
    weights = render.integrate_weights(density, delta_t)
    a = torch.sum(weights, dim=1)
    result = torch.sum(rgb * weights, dim=1)
    if config['render']['white_bkgd']:
        result = result + (1 - a)
    return weights, result

def train(train_dataset, nerf_model, encoder, dir_encoder):
    opt = torch.optim.AdamW(nerf_model.parameters(), lr=args.lr)
    conf_train = config['train']
    conf_render = config['render']
    def lr_lambda(i):
        lr_init = conf_train['lr']['init']
        lr_final = conf_train['lr']['final']
        if i < conf_train['warmup']:
            ret = lr_init + (lr_init - lr_final) * np.sin(np.pi / 2 * i / conf_train['warmup'])
        else:
            ret = 1
            num_iter = args.epoch * len(train_dataset) // conf_train['batch']
            w = i / num_iter
            ret *= np.exp((1-w) * np.log(lr_init) + w * np.log(lr_final))
        return ret / args.lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if args.from_ckpt:
        checkpoint = torch.load(config['path']['ckpt'])
        nerf_model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        sch.load_state_dict(checkpoint['sch'])
    
    loss_func = nn.MSELoss()

    dataloader = DataLoader(train_dataset, 
                            batch_size=conf_train['batch'],
                            shuffle=True,
                            num_workers=conf_train['num_workers'],
                            pin_memory=True)
    scaler = torch.cuda.amp.GradScaler()

    nerf_model.train()
    with torch.autograd.set_detect_anomaly(False):
        for epoch in range(checkpoint['epoch'] if conf_train['from_ckpt'] else 0, conf_train['epoch']):
            epoch_loss = 0
            pbar = tqdm.tqdm(dataloader)
            for it, (o, d, ground_truth, pixel_size, scale) in enumerate(pbar):
                with torch.cuda.amp.autocast(enabled=True):
                    opt.zero_grad()

                    o = o.to(device)
                    d = d.to(device)
                    ground_truth = ground_truth.to(device)
                    if conf_render['white_bkgd']:
                        ground_truth = ground_truth[..., :3] * ground_truth[..., -1:] + (1 - ground_truth[..., -1:])
                    else:
                        ground_truth = ground_truth[..., :3] * ground_truth[..., -1:]
                    r = pixel_size * 2 / np.sqrt(12) # sqrt(3)
                    r = r.to(device, torch.float32)
                    o = o[:, None, :]
                    d = d[:, None, :]
                    r = r[:, None, None]

                    t_vals = render.gen_intervals(
                        conf_render['near'], 
                        conf_render['far'], o.shape[0],
                        conf_render['samples']['coarse'],
                        device=device)
                    weights_coarse, result_coarse = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

                    fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
                    fine_t_vals = fine_sampler.sample(
                        conf_render['samples']['fine'] + 1,
                        random=True,
                        stratified=False).detach()

                    _, result_fine = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder)

                    s = scale[:, None].to(device)
                    coarse_loss = loss_func(result_coarse * s, ground_truth * s)
                    fine_loss = loss_func(result_fine * s, ground_truth * s)
                    coarse_estimated_loss = loss_func(result_coarse, ground_truth)
                    fine_estimated_loss = loss_func(result_fine, ground_truth)
                    loss = fine_loss + conf_train['lambda_coarse'] * coarse_loss
                    epoch_loss += loss

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sch.step()
                    alpha = torch.sum(weights_coarse, dim=1)
                    nan_grad_count = 0
                    for p in nerf_model.parameters():
                        nan_grad_count += torch.sum(torch.isnan(p.grad))
                summary = [
                    f'epoch {epoch}',
                    f'coarse/fine/loss {coarse_estimated_loss:.3e}/{fine_estimated_loss:.3e}/{loss:.3e}',
                    f'alpha {torch.max(alpha):.3f}/{torch.min(alpha):.3f}',
                    f'nan {nan_grad_count}'
                ]
                pbar.set_description(' | '.join(summary))
            torch.save({
                'model': nerf_model.state_dict(),
                'opt': opt.state_dict(),
                'sch': sch.state_dict(),
                'epoch': epoch
                }, config['path']['ckpt'])
            print(f'avg training loss: {(epoch_loss/len(dataloader)):.8e}')

def test(test_dataset, nerf_model, encoder, dir_encoder):
    conf_render = config['render']
    dataloader = DataLoader(test_dataset, 
                            batch_size=test_dataset.widths[0],
                            shuffle=False)
    nerf_model.load_state_dict(torch.load(config['path']['ckpt'])['model'])
    nerf_model.eval()
    with torch.no_grad():
        rows = []
        depth_rows = []
        idx_png = 0
        for o, d, ground_truth, pixel_size, scale in tqdm.tqdm(dataloader):
            o = o.to(device)
            d = d.to(device)
            r = pixel_size / np.sqrt(3) # sqrt(3)
            r = r.to(device, torch.float32)
            o = o[:, None, :]
            d = d[:, None, :]
            r = r[:, None, None]

            t_vals = render.gen_intervals(
                conf_render['near'],
                conf_render['far'],
                test_dataset.widths[0],
                conf_render['samples']['coarse'],
                device=device)

            weights_coarse, _ = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

            fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
            fine_t_vals = fine_sampler.sample(
                conf_render['samples']['fine'] + 1,
                stratified=True,
                random=True)

            weights_fine, result_fine = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder)
            result_fine = torch.cat([result_fine, torch.ones(result_fine.shape[0], 1, device=device)], dim=-1)
            E_t_vals = (fine_t_vals[:, :-1, :] + fine_t_vals[:, 1:, :]) / 2
            sum_w = torch.sum(weights_fine, dim=1)
            depths = torch.sum(weights_fine * E_t_vals, dim=1) / sum_w
            depths = (depths - conf_render['near']) / (conf_render['far'] - conf_render['near'])
            depths = depths * -0.8 + 0.9
            depths = depths.expand(-1, 3)
            depths = torch.cat([depths, torch.ones(depths.shape[0], 1, device=device)], dim=-1)

            rows.append(result_fine)
            depth_rows.append(depths)
            if len(rows) % test_dataset.heights[0] == 0:
                img = torch.stack(rows)
                img = img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGBA')
                rows = []
                img.save(f'output/{idx_png}.png')

                img = torch.stack(depth_rows)
                print(img.shape)
                img = img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGBA')
                depth_rows = []
                img.save(f'output/{idx_png}_d.png')
                idx_png += 1


if __name__ == "__main__":
    encoder = model.IntegratedPositionalEncoder(
        freq_range=[config['render']['freq']['low'], config['render']['freq']['high']]
    ).to(device)
    dir_encoder = model.PositionalEncoder(
        input_size=2,
        freq_range=[config['render']['freq_dir']['low'], config['render']['freq_dir']['high']]
        ).to(device)
    nerf_model = model.NerfModel(input_dim=encoder.output_size(), 
                                input_dim_dir=dir_encoder.output_size()).to(device).float()
    raw_data = dataset.load_synthetic(config['path']['data'], verbose=True)
    
    if args.mode == 'train':
        train_dataset = dataset.PixelDataset(raw_data['train'], mip_level=config['render']['mip'])
        train(train_dataset, nerf_model, encoder, dir_encoder)
    elif args.mode == 'test':
        test_dataset = dataset.PixelDataset(raw_data['test'], mip_level=1)
        test(test_dataset, nerf_model, encoder, dir_encoder)
