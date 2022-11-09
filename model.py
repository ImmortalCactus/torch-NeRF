import torch.nn as nn
import torch

class PositionalEncoder(nn.Module):
    def __init__(self, input_size=3,
                encoding_functions=[torch.sin, torch.cos],
                freq_range=(-2, 8),
                keep_raw=False,
                multiply_by_pi=False):
        super().__init__()
        self.input_size = input_size
        self.freq_range = freq_range
        self.encoding_functions = encoding_functions
        self.keep_raw = keep_raw
        self.multiply_by_pi = multiply_by_pi
    
    def forward(self, x):
        log_freqs = torch.arange(*self.freq_range)
        log_freqs = log_freqs.to(x.device)
        freqs = torch.pow(2, log_freqs)
        freqs = torch.broadcast_to(freqs, (*x.shape[:-1], 1, freqs.shape[0]))
        raw = torch.unsqueeze(x, len(x.shape))
        if self.multiply_by_pi:
            raw = raw * torch.pi
        pre_f = torch.reshape(freqs * raw, [*x.shape[:-1], -1])

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
    
    def forward(self, o, d, r, t0, t1):
        t_mu = (t0 + t1) / 2 # [1, samples_on_ray, 1]
        t_delta = (t1 - t0) / 2 # [1, samples_on_ray, 1]
        t_mu2 = t_mu * t_mu # [1, samples_on_ray, 1]
        t_delta2 = t_delta * t_delta # [1, samples_on_ray, 1]
        t_delta4 = t_delta2 * t_delta2 # [1, samples_on_ray, 1]
        A = 2 * t_mu * t_delta2 # [1, samples_on_ray, 1]
        B = 3 * t_mu2 + t_delta2 # [1, samples_on_ray, 1]
        mu_t = t_mu + A / B # [1, samples_on_ray, 1]
        sigma_t2 = t_delta2 / 3 - 4 * t_delta4 * (12 * t_mu2 - t_delta2) / (15 * B * B) # [1, samples_on_ray, 1]
        sigma_r2 = r * r * (t_mu2 / 4 + 5 * t_delta2 / 12 - 4 * t_delta4 / (15 * B)) # [1, samples_on_ray, 1]
        
        mu = o + mu_t * d # [batch, samples_on_ray, 3]
        dd = d * d # [batch, 1, 3]
        cov_diag = sigma_t2 * dd + sigma_r2 * (1 - dd / torch.sum(dd, dim=-1, keepdim=True))
        
        log_freqs = torch.arange(*self.freq_range).to(o.device)
        freqs = torch.pow(4, log_freqs)
        freqs = freqs.reshape([1, 1, -1, 1])
        cov_diag_gamma = freqs * torch.unsqueeze(cov_diag, 2)
        cov_diag_gamma = cov_diag_gamma.reshape(*cov_diag_gamma.shape[:-2], -1)

        mu_gamma = freqs * torch.unsqueeze(mu, 2)
        mu_gamma = mu_gamma.reshape(*mu_gamma.shape[:-2], -1)

        encoded = [torch.sin(mu_gamma) * torch.exp(- cov_diag_gamma / 2),
                    torch.cos(mu_gamma) * torch.exp(- cov_diag_gamma / 2)]

        encoded = torch.cat(encoded, dim=-1)
        return encoded
    
    def output_size(self):
        return 3 * 2 * (self.freq_range[1] - self.freq_range[0])

class shifted_soft_plus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.log(1 + torch.exp(x-1))

class widened_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (1 + 2 * 0.001) / (1 + torch.exp(-x)) - 0.001

class NerfModel(nn.Module):
    def __init__(self, input_dim=69, 
                    input_dim_dir=22, 
                    hidden_dim=256, 
                    num_pre_res=4, 
                    num_post_res=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.pre_res_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for l in range(num_pre_res)])
        self.residual_layer = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.post_res_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for l in range(num_post_res)])
        self.density_output_layer = nn.Linear(hidden_dim, hidden_dim+1)
        self.bottleneck_layer = nn.Linear(hidden_dim + 1 + input_dim_dir, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 3)
        self.act_func = nn.ReLU()
        self.output_act_func = nn.Sigmoid()
    
    def forward(self, input_pos, input_dir):
        output = self.act_func(self.input_layer(input_pos))
        for l in self.pre_res_layers:
            output = self.act_func(l(output))
        output = torch.cat([output, input_pos], axis=-1)
        output = self.act_func(self.residual_layer(output))
        for l in self.post_res_layers:
            output = self.act_func(l(output))
        output = self.act_func(self.density_output_layer(output))
        density = output[..., -1]
        input_dir = input_dir.expand(-1, output.shape[1], -1)
        output = torch.cat([output, input_dir], dim=-1)
        output = self.act_func(self.bottleneck_layer(output))
        rgb = self.output_act_func(self.output_layer(output))
        return density, rgb

def integrate(sigma, c, delta):
    # [batch, sample per ray, 3/1]
    T = sigma * delta
    T = torch.cumsum(T, dim=1).roll(1, -1)
    T[:, 0, :] *= 0
    T = -T
    T = torch.exp(T)

    alpha = torch.ones([c.shape[0], c.shape[1], 1]).to(c.device)
    c = torch.cat([c, alpha], dim=-1)
    T = T * c
    T = T * (1 - torch.exp(- sigma * delta))
    C = torch.sum(T, dim=1)

    return C