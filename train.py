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
parser.add_argument('--config', type=str, default='config.yml')
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

if torch.cuda.is_available():
    device = torch.device(config['resource']['device'])
else:
    device = torch.device('cpu')
print(f'using {device} as device')

def forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder, more_info=False):
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
    if more_info:
        t_mid = (t_vals[:, 1:, :] + t_vals[:, :-1, :]) / 2
        depths = torch.sum(t_mid * weights, dim=1)
        return weights, result, a, depths
    else:
        return weights, result

class MipLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(MipLRDecay, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]

def train(train_dataset, nerf_model, encoder, dir_encoder):
    conf_train = config['train']
    conf_render = config['render']
    opt = torch.optim.AdamW(nerf_model.parameters(),
                            lr=conf_train['lr']['init'],
                            weight_decay=conf_train['weight_decay'])
    
    sch = MipLRDecay(
        opt,
        conf_train['lr']['init'],
        conf_train['lr']['final'],
        conf_train['epoch'] * len(train_dataset) // conf_train['batch'],
        lr_delay_steps=conf_train['warmup'],
        lr_delay_mult=conf_train['warmup_mult']
    )

    if conf_train['from_ckpt']:
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
    with torch.autograd.set_detect_anomaly(conf_train['detect_anomaly']):
        for epoch in range((checkpoint['epoch']+1) if conf_train['from_ckpt'] else 0, conf_train['epoch']):
            epoch_loss = 0
            pbar = tqdm.tqdm(dataloader)
            for o, d, ground_truth, pixel_size, scale in pbar:
                with torch.cuda.amp.autocast(enabled=conf_train['mixed_precision']):

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
                        conf_render['far'],
                        o.shape[0],
                        conf_render['samples']['coarse'],
                        device=device)
                    weights_coarse, result_coarse = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

                    fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
                    fine_t_vals = fine_sampler.sample(
                        conf_render['samples']['fine'] + 1,
                        random=True
                    ).detach()

                    _, result_fine = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder)

                    s = scale[:, None].to(device)
                    coarse_loss = loss_func(result_coarse * s, ground_truth * s)
                    fine_loss = loss_func(result_fine * s, ground_truth * s)
                    coarse_estimated_loss = loss_func(result_coarse, ground_truth)
                    fine_estimated_loss = loss_func(result_fine, ground_truth)
                    loss = fine_loss + conf_train['lambda_coarse'] * coarse_loss
                    epoch_loss += loss

                    opt.zero_grad()
                    if conf_train['mixed_precision']:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    sch.step()
                    alpha = torch.sum(weights_coarse, dim=1)
                summary = [
                    f'epoch {epoch}',
                    f'coarse/fine/loss {coarse_estimated_loss:.3e}/{fine_estimated_loss:.3e}/{loss:.3e}',
                    f'alpha {torch.max(alpha):.3f}/{torch.min(alpha):.3f}'
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
                            batch_size=config['train']['batch'],
                            shuffle=False)
    nerf_model.load_state_dict(torch.load(config['path']['ckpt'])['model'])
    nerf_model.eval()
    with torch.no_grad():
        pixels = torch.zeros([0, 3], device=device)
        alphas = torch.zeros([0, 1], device=device)
        depths = torch.zeros([0, 1], device=device)
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
                o.shape[0],
                conf_render['samples']['coarse'],
                device=device)

            weights_coarse, _ = forward(o, d, r, t_vals, nerf_model, encoder, dir_encoder)

            fine_sampler = render.FineSampler2(weights_coarse, t_vals, alpha=0.001)
            fine_t_vals = fine_sampler.sample(
                conf_render['samples']['fine'] + 1,
                random=True)

            weights_fine, result_fine, a, d = forward(o, d, r, fine_t_vals, nerf_model, encoder, dir_encoder, more_info=True)

            pixels = torch.cat([pixels, result_fine])
            alphas = torch.cat([alphas, a])
            depths = torch.cat([depths, d])
            img_size = test_dataset.widths[0] * test_dataset.heights[0]
            while pixels.shape[0] >= img_size:
                img = pixels[:img_size, :]
                pixels = pixels[img_size:, :]
                img = img.reshape([test_dataset.widths[0], test_dataset.heights[0], 3])
                img = img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGB')
                img.save(f'output/{idx_png}.png')

                t_normalized = (depths[:img_size, :] - conf_render['near']) / (conf_render['far'] - conf_render['near'])
                d_img = torch.cat([t_normalized.expand([-1, 3]), alphas[:img_size, :]], dim=1)
                d_img = d_img.reshape([test_dataset.widths[0], test_dataset.heights[0], 4])
                img = d_img.cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGBA')
                img.save(f'output/d_{idx_png}.png')

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
