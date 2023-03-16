import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")


        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # z_vals = None
        near = self.min_depth
        far = self.max_depth
        npts = self.n_pts_per_ray
        z_vals = torch.linspace(near, far, npts).to(device)

        # TODO (1.4): Sample points from z values
        sample_points = None

        ndirs = ray_bundle.origins.shape[0]
        # num_pts = z_vals.shape[0]
        sample_points = torch.zeros(ndirs, npts, 3).to(device)
        for i in range(npts):
            sample_points[:,i,:] = ray_bundle.origins + (z_vals[i] * ray_bundle.directions)
        
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths = (z_vals * torch.ones_like(sample_points[:,0,:1])).unsqueeze(2)
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}