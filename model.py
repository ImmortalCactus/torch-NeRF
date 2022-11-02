import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, encoding_functions=[torch.sin, torch.cos]):
        super().__init__()
        self.encoding_functions = encoding_functions
    
    def forward(self, x, max_log_freq, keep_raw=False):
        log_freqs = torch.arange(0, max_log_freq+1)
        freqs = torch.pow(2, log_freqs)
        freqs = torch.broadcast_to(freqs, (*x.shape[:-1], 1, freqs.shape[0]))
        raw = torch.unsqueeze(x, len(x.shape))
        pre_f = torch.reshape(freqs * raw, [*x.shape[:-1], -1])

        if keep_raw: encoded = x
        for f in self.encoding_functions:
            post_f = f(pre_f)
            encoded = torch.cat([encoded, post_f], dim=-1)
        
        return encoded

def shifted_soft_plus(x):
    return torch.log(1 + torch.exp(x-1))

def widened_sigmoid(x, epsilon=0.001):
    return (1 + 2 * epsilon) / (1 + torch.exp(-x)) - epsilon

class NerfModel(nn.module):
    def __init__(self, input_dim=69, 
                    input_dim_dir=27, 
                    hidden_dim=256, 
                    num_pre_res=4, 
                    num_post_res=2, 
                    act_func=shifted_soft_plus, 
                    output_act_func=widened_sigmoid):
        self.input_layer = nn.linear(input_dim, hidden_dim)
        self.pre_res_layers = nn.ModuleList(
            [nn.linear(hidden_dim, hidden_dim)] for l in range(num_pre_res))
        self.residual_layer = nn.linear(hidden_dim + input_dim, hidden_dim)
        self.post_res_layers = nn.ModuleList(
            [nn.linear(hidden_dim, hidden_dim)] for l in range(num_post_res))
        self.desity_output_layer = nn.linear(hidden_dim, hidden_dim+1)
        self.bottleneck_layer = nn.linear(hidden_dim + 1 + input_dim_dir, hidden_dim // 2)
        self.output_layer = nn.linear(hidden_dim // 2, 3)
        self.act_func = act_func
    
    def forward(input_pos, input_dir):
        output = self.act_func(self.input_layer(input_pos))
        for l in self.pre_res_layers:
            output = self.act_func(l(output))
        output = torch.cat([output, input_pos], axis=-1)
        output = self.act_func(self.residual_layer(output))
        for l in self.post_res_layers:
            output = self.act_func(l(output))
        output = self.act_func(self.density_output_layer(output))
        output = torch.cat([output, input_dir], axis=-1)
        output = self.act_func(self.bottleneck_layer(output))
        output = widened_sigmoid(self.output_layer(output))

def sample_rays(model, encoder, ):
