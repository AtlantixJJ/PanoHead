# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import trimesh
import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


def unproject_depth(depth, cam2world_matrix, intrinsics, resolution):
    """
    Create batches of rays and return origins and directions.

    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    resolution: int

    ray_origins: (N, M, 3)
    ray_dirs: (N, M, 2)
    """
    N, M = cam2world_matrix.shape[0], resolution**2
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = torch.ones((N, M), device=cam2world_matrix.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

    world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--pose_cond', type=int, help='camera conditioned pose angle', default=90, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    pose_cond: int,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained model.
    python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 \
        --network models/easy-khair-180-gpc0.8-trans10-025000.pkl
    """
    torch.set_grad_enabled(False)
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        #G_dict = legacy.load_network_pkl(f)['G_ema'].state_dict()
    #c = json.load("training_options.json")
    #common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=3)
    #G = dnnlib.util.construct_class_by_name(**c['G_kwargs'], **common_kwargs).eval().#requires_grad_(False).to(device)
    #G.load_state_dict(G_dict)
    #del G_dict

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    network_pkl = os.path.basename(network_pkl)
    outdir = os.path.join(outdir, os.path.splitext(network_pkl)[0] + '_' + str(pose_cond))
    os.makedirs(outdir, exist_ok=True)

    pose_cond_rad = pose_cond/180*np.pi

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # cond camera settings
        cam_pivot = torch.tensor([0, 0, 0], device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        # z and w
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        # z1 = torch.from_numpy(np.random.RandomState(44).randn(1, G.z_dim)).to(device)
        # ws_list = []
        # ws_list.append(G.mapping(torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device), conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff))
        # ws_list.append(G.mapping(torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device), conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff))
        # ws_list.append(G.mapping(z1, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff))
 
        imgs, point_images, point_cloud_pos, point_cloud_color = [], [], [], []
        angle_p = -0.2
        # for angle_y, angle_p in [(2.1, angle_p), (1.05, angle_p), (0, angle_p), (-1.05, angle_p), (-2.1, angle_p)]:
        for idx, angles in enumerate([(0, angle_p), (-45/180*np.pi, angle_p), (-90/180*np.pi, angle_p), (-135/180*np.pi, angle_p), (-np.pi, angle_p), (-225/180*np.pi, angle_p)]):
            angle_y = angles[0]
            angle_p = angles[1]
            # rand camera setting
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            
            # img = G.synthesis(ws, camera_params, ws_bcg = ws_list[idx])['image']
            res = G.synthesis_point(ws, camera_params)
            img = res['image'].permute(0, 2, 3, 1).contiguous().cpu() # (N, H, W, C)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            points = torch.nn.functional.interpolate(
                res['avg_pos'], img.shape[1:3], mode="bilinear")
            points = points.permute(0, 2, 3, 1).contiguous().cpu()
            point_cloud_pos.append(points.view(-1, 3).clone())
            point_cloud_color.append(img.view(-1, 3).clone())
            
            point_img = (points - points.min()) / (points.max() - points.min())
            point_img = (point_img * 255).to(torch.uint8)
            imgs.append(img)
            point_images.append(point_img)

        img = torch.cat(imgs, dim=2)
        point_image = torch.cat(point_images, dim=2)
        point_cloud_pos = torch.cat(point_cloud_pos)
        point_cloud_color = torch.cat(point_cloud_color)
        pc_obj = trimesh.PointCloud(point_cloud_pos, point_cloud_color)
        pc_obj.export(f'{outdir}/point_seed{seed:04d}.ply')

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        PIL.Image.fromarray(point_image[0].cpu().numpy(), 'RGB').save(f'{outdir}/pointimage_seed{seed:04d}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
