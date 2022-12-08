import argparse
import tqdm
import imageio.v3 as imageio
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='./nerf_synthetic/lego/test')
    parser.add_argument('--res', type=str, default='./output')
    parser.add_argument('--white_bkgd', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    psnr_sum = 0

    for i in tqdm.trange(200):
        ref = imageio.imread(f'{args.ref}/r_{i}.png')
        ref = ref / 255.
        if args.white_bkgd:
            ref = ref[..., :3] * ref[..., 3:] + (1 - ref[..., 3:])
        else:
            ref = ref[..., :3] * ref[..., 3:]
        res = imageio.imread(f'{args.res}/{i}.png')
        res = res / 255.
        res = res[..., :3]

        diff = ref - res
        diff = diff**2
        diff = np.sum(diff, axis=-1)
        diff = np.mean(diff)
        psnr = -10 * (np.log(diff) / np.log(10))
        psnr_sum += psnr
    
    print(psnr_sum/200)