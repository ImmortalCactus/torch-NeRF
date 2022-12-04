# A PyTorch Implementation of Mip-NeRF
This is a PyTorch implementation of Mip-NeRF by Barron et al. This is a work in progress.
# Run
Training 

`$ python train.py --config=config.yml`

Currently only the Blender synthetic dataset is supported. Which can be downloaded from the link on the original NeRF repo.

Testing 

`$ python train.py --config=config.yml --mode=test`

# Results
~20 epochs on the LEGO dataset with no multiscale
![20 epoch result](./misc/20.gif)
![20 epoch depth](./misc/d20.gif)
# References
* [Original Paper](https://arxiv.org/abs/2103.13415)
* [mipnerf-pytorch](https://github.com/bebeal/mipnerf-pytorch/tree/7b2ec348093285e1f549c52d54fb3871b987e6f5)