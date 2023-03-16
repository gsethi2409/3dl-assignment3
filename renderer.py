import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # TODO (1.5): Compute transmittance using the equation described in the README
        # pass
        ndirs = rays_density.shape[0]
        npts = rays_density.shape[1]

        sigmas = [torch.ones((ndirs, 1)).to(device)] # T = 1

        for pt in range(npts - 1):
            e = torch.exp(-rays_density[:, pt] * deltas[:, pt])
            sigmas.append(sigmas[-1] * e)

        stacked_sigmas = torch.stack(sigmas, dim=1)
        
        # TODO (1.5): Compute weight used for rendering from transmittance and density
        weights = stacked_sigmas * (1 - torch.exp(-rays_density * deltas))
        
        return weights

    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # pass

        try:
            weighted_features = (weights * rays_feature)
        except:
            weighted_features = (weights.squeeze(2) * rays_feature)
            
        
        feature = torch.sum(weighted_features, axis=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # pass
            feature = self._aggregate(weights, feature.reshape(-1,64,3))
            # feature = self._aggregate(weights, feature)


            # TODO (1.5): Render depth map
            # pass
            depth = self._aggregate(weights, depth_values)
            # depth = self._aggregate_depths(weights, depth_values)


            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
