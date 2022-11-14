import numpy as np
import torch
def get_c2w(t, rho_x, rho_y):
    translation = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(-rho_x), -np.sin(-rho_x), 0],
        [0, np.sin(-rho_x), np.cos(-rho_x), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    rotation_y = np.array([
        [np.cos(rho_y), 0, np.sin(rho_y), 0],
        [0, 1, 0, 0],
        [-np.sin(rho_y), 0, np.cos(rho_y), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    swap = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return swap @ rotation_y @ rotation_x @ translation


def get_rays(h, w, f, matrices):
    x_axis = (np.arange(w, dtype=np.float32) - w/2 + 0.5) / f
    y_axis = (np.arange(h, dtype=np.float32) - h/2 + 0.5) / f * -1
    x, y = np.meshgrid(x_axis, y_axis, indexing='xy')
    z = np.ones_like(x, dtype=np.float32) * -1
    
    coords = np.stack([x,y,z], axis=-1)
    coords = coords.reshape([-1, 3])
    coords = np.broadcast_to(coords, [matrices.shape[0], *coords.shape])

    origin = matrices[:, :3, -1]
    dirs = np.matmul(coords, matrices[:, :3, :3].transpose((0, 2, 1)))
    dirs = dirs.reshape(matrices.shape[0], h, w, 3)
    return origin.astype(np.float32), dirs.astype(np.float32)

def get_spherical(d):
    theta = torch.arccos(d[..., 2])
    c = torch.complex(d[..., 0], d[..., 1])
    phi = torch.angle(c)
    return torch.stack([theta, phi], axis=-1)

def integrate_weights(density, delta):
    # [batch, sample per ray, 3/1]
    T = density * delta
    T = torch.cumsum(T, dim=1).roll(1, 1)
    T[:, 0, :] *= 0
    T = -T
    T = torch.exp(T)
    T = T * (1 - torch.exp(- density * delta))
    return T

def gen_intervals(near, far, batch_size, num_sample):
    """generate intervals to encoded upon"""
    t = torch.linspace(near, far, num_sample+1)
    mid = (t[:-1] + t[1:]) / 2
    lower = torch.cat([t[:1], mid], dim=-1).unsqueeze(0)
    upper = torch.cat([mid, t[-1:]], dim=-1).unsqueeze(0)
    random_w = torch.rand(batch_size, num_sample+1)
    t = lower * (1 - random_w) + upper * random_w
    t = torch.unsqueeze(t, 2)
    return t

class FineSampler():
    def __init__(self, weights, t_vals, alpha=0):
        self.dist = torch.distributions.categorical.Categorical((weights+alpha).reshape(*weights.shape[:2]))
        self.t_vals = t_vals
    def sample(self, n):
        sampled = self.dist.sample((n,))
        sampled = torch.swapaxes(sampled, 0, 1)
        upper = []
        lower = []
        for i in range(sampled.shape[0]):
            lower.append(torch.index_select(self.t_vals[i], 0, sampled[i]))
            upper.append(torch.index_select(self.t_vals[i], 0, sampled[i]+1))
        upper = torch.stack(upper, dim=0)
        lower = torch.stack(lower, dim=0)
        w = torch.rand(*upper.shape).to(upper.device)
        ret = lower * (1-w) + upper * w
        ret, _ = ret.sort(dim=1)
        return ret

class FineSampler2():
    def __init__(self, weights, t_vals, alpha=0):
        self.weights = weights + alpha
        self.weights = torch.cat([self.weights[:, :1, :], self.weights, self.weights[:, -1:, :]], dim=1)
        self.weights = torch.maximum(self.weights[:, 1:, :], self.weights[:, :-1, :])
        self.weights = (self.weights[:, :-1, :] + self.weights[:, 1:, :]) / 2
        self.cum_w = torch.cumsum(self.weights, dim=1)
        self.cum_w = self.cum_w / self.cum_w[:, -1:, :]
        self.cum_w = torch.cat([torch.zeros(self.cum_w.shape[0], 1, 1, device=self.cum_w.device), self.cum_w], dim=1)
        self.t_vals = t_vals

        self.cum_w = self.cum_w.swapaxes(1, 2) # [batch, fine, coarse]
        self.t_vals = self.t_vals.swapaxes(1, 2)
    def sample(self, n, random=True):
        sampled = torch.linspace(0, 1-1e-10, n, device=self.cum_w.device)
        if random:
            mid = (sampled[:-1] + sampled[1:]) / 2
            lower = torch.cat([sampled[:1], mid], dim=-1).unsqueeze(0)
            upper = torch.cat([mid, sampled[-1:]], dim=-1).unsqueeze(0)
            random_w = torch.rand(self.cum_w.shape[0], n, device=self.cum_w.device)
            sampled = lower * (1 - random_w) + upper * random_w
            sampled = sampled.unsqueeze(2)
        else:
            sampled = sampled.reshape([1, -1, 1])
        
        mask = sampled >= self.cum_w
        
        lower_w, _ = torch.max(torch.where(mask, self.cum_w, self.cum_w[:, :, :1]), dim=2, keepdim=True)
        upper_w, _ = torch.min(torch.where(~mask, self.cum_w, self.cum_w[:, :, -1:]), dim=2, keepdim=True)
        lower_t, _ = torch.max(torch.where(mask, self.t_vals, self.t_vals[:, :, :1]), dim=2, keepdim=True)
        upper_t, _ = torch.min(torch.where(~mask, self.t_vals, self.t_vals[:, :, -1:]), dim=2, keepdim=True)

        w = (sampled - lower_w) / (upper_w - lower_w + 1e-10)
        return lower_t + w * (upper_t - lower_t)