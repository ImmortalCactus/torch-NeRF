import torch.nn as nn
import torch

class PositionalEncoder(nn.Module):
    def __init__(self, input_size=3,
                encoding_functions=[torch.sin, torch.cos],
                freq_range=(-2, 8),
                keep_raw=False):
        super().__init__()
        self.input_size = input_size
        self.freq_range = freq_range
        log_freqs = torch.arange(*self.freq_range)
        freqs = torch.pow(2, log_freqs)
        self.freqs = freqs.reshape([1, 1, -1, 1])
        self.register_buffer('freq_const', self.freqs)
        self.encoding_functions = encoding_functions
        self.keep_raw = keep_raw
    
    def forward(self, x):
        raw = x[..., None, :]
        pre_f = self.freq_const * raw
        pre_f = pre_f.reshape([*pre_f.shape[:-2], -1])

        if self.keep_raw:
            encoded = [x]
        else:
            encoded = []
        for f in self.encoding_functions:
            encoded.append(f(pre_f))
        
        encoded = torch.cat(encoded, dim=-1)
        return encoded
    
    def output_size(self):
        return len(self.encoding_functions) * self.input_size * (self.freq_range[1]-self.freq_range[0]) \
                + self.keep_raw * self.input_size

class IntegratedPositionalEncoder(nn.Module):
    def __init__(self, freq_range=(-5, 5)):
        super().__init__()
        self.freq_range = freq_range
        log_freqs = torch.arange(*self.freq_range)
        freqs = torch.pow(4, log_freqs)
        self.freqs = freqs.reshape([1, 1, -1, 1])
        self.register_buffer('freq_const', self.freqs)
    
    def forward(self, o, d, r, t0, t1):
        t_mu = (t0 + t1) / 2 # [1, samples_on_ray, 1]
        t_delta = (t1 - t0) / 2 # [1, samples_on_ray, 1]
        mu_t = t_mu + (2 * t_mu * t_delta**2) / (3 * t_mu**2 + t_delta**2)
        var_t = (t_delta**2) / 3 - (4 / 15) * ((t_delta**4 * (12 * t_mu**2 - t_delta**2)) / (3 * t_mu**2 + t_delta**2)**2)
        var_r = r**2 * ((t_mu**2) / 4 + (5/12) * t_delta**2 - 4/15 * (t_delta**4) / (3 * t_mu**2 + t_delta**2)) # [1, samples_on_ray, 1]
        
        mu = o + mu_t * d # [batch, samples_on_ray, 3]
        dd = d**2 # [batch, 1, 3]
        mag = torch.sum(dd, dim=-1, keepdim=True)
        cov_diag = var_t * dd + var_r * (1 - dd / mag)
        
        cov_diag_gamma = self.freq_const * cov_diag[..., None, :]
        cov_diag_gamma = cov_diag_gamma.reshape(*cov_diag_gamma.shape[:-2], -1)

        mu_gamma = self.freq_const * mu[..., None, :]
        mu_gamma = mu_gamma.reshape(*mu_gamma.shape[:-2], -1)

        encoded = [torch.sin(mu_gamma) * torch.exp(- cov_diag_gamma / 2),
                    torch.cos(mu_gamma) * torch.exp(- cov_diag_gamma / 2)]

        encoded = torch.cat(encoded, dim=-1)
        return encoded
    
    def output_size(self):
        return 3 * 2 * (self.freq_range[1] - self.freq_range[0])

class NerfModel(nn.Module):
    def __init__(self, input_dim=69, 
                    input_dim_dir=22,
                    hidden_dim=256):
        super().__init__()
        self.density0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.density1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.density_out = nn.Linear(hidden_dim, 1)
        self.rgb0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rgb1 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim_dir, hidden_dim),
            nn.ReLU(True)
        )
        self.rgb_out = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
    
    def forward(self, input_pos, input_dir):
        output = self.density0(input_pos)
        output = torch.cat([output, input_pos], axis=-1)
        output = self.density1(output)
        density = self.density_out(output)
        output = self.rgb0(output)
        output = torch.cat([output, input_dir.expand(-1, output.shape[1], -1)], dim=-1)
        output = self.rgb1(output)
        rgb = self.rgb_out(output)
        return density, rgb
