import argparse

import imageio.v3 as imageio
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str)
    parser.add_argument('--res', type=str)
    parser.add_argument('--out', type=str, default='test.png')
    parser.add_argument('--white_bkgd', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    ref = imageio.imread(args.ref)
    ref = ref / 255.
    if args.white_bkgd:
        ref = ref[..., :3] * ref[..., 3:] + (1 - ref[..., 3:])
    else:
        ref = ref[..., :3] * ref[..., 3:]
    res = imageio.imread(args.res)
    res = res / 255.
    res = res[..., :3]

    diff = ref - res
    # diff = diff ** 2
    # diff = np.mean(diff, axis=-1, keepdims=True)
    # diff = np.ones_like(res) * diff
    #diff = diff / 2. + 0.5
    diff = np.maximum(diff, 0)

    diff = diff * 255
    diff = diff.astype(np.uint8)
    img = Image.fromarray(diff, 'RGB')
    img.save(args.out)

    