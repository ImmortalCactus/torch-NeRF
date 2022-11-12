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

def get_ray(h, w, f, matrix, i, j):
    x = (j + 0.5 - h/2) / f
    y = (i + 0.5 - w/2) * -1 / f
    z = -1

    coord = np.array([x, y, z])
    coord = coord / np.linalg.norm(coord)

    o = matrix[:3,-1]
    d = np.matmul(coord, matrix[:3, :3].T)
    return o, d

def get_spherical(d):
    theta = torch.arccos(d[..., 2])
    c = torch.complex(d[..., 0], d[..., 1])
    phi = torch.angle(c)
    return torch.stack([theta, phi], axis=-1)