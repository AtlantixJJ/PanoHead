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
import math
import legacy
import click
import dnnlib
import numpy as np
import torch
import kaolin as kal
from skimage.io import imsave
from skimage.measure import marching_cubes
from typing import List, Optional, Tuple, Union
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


def normalize(x, mode="[0,1]"):
    xmin, xmax = x.min(), x.max()
    if mode == "[0,1]":
        return (x - xmin) / (xmax - xmin)
    

def visualize_depth(fg_mask, depth):
    """Visualize the rasterized depth."""
    fg_depth = depth[fg_mask.bool()]
    zmin, zmax = fg_depth.min(), fg_depth.max()
    depth_disp = (depth - zmin) / (zmax - zmin) * fg_mask
    return (depth_disp * 255).byte()

#----------------------------------------------------------------------------

def rasterize_mesh(mesh, cam, component="depth,normal"):
    """Rasterize a mesh with a given camera.
    """

    # vertices in camera coordinate
    v_cam = cam.extrinsics.transform(mesh.vertices)
    # vertices in image coordinate
    v_ndc = cam.intrinsics.transform(v_cam)
    # face vertices in camera coordinate
    fv_cam = kal.ops.mesh.index_vertices_by_faces(
        v_cam, mesh.faces)
    # face vertices in image coordinate
    fv_ndc = kal.ops.mesh.index_vertices_by_faces(
        v_ndc[..., :2], mesh.faces)
    
    # features to be rasterized
    face_features, names = [], []
    if "depth" in component:
        face_features.append(fv_cam[0, ..., -1:])
        names.append("depth")
    if "normal" in component:
        face_features.append(mesh.face_normals)
        names.append("normal")

    im_features, face_idx  = kal.render.mesh.rasterize(
        height=cam.height, width=cam.width,
        face_vertices_z=fv_cam[..., -1],
        face_vertices_image=fv_ndc,
        face_features=face_features)

    # foreground mask
    res_dic = {"fg_mask": face_idx != -1}

    for idx, name in enumerate(names):
        res_dic[name] = im_features[idx]

    return res_dic

#----------------------------------------------------------------------------

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default="0-3")
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, default="out", metavar='DIR')
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

    print('=> Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    network_pkl = os.path.basename(network_pkl)
    outdir = os.path.join(outdir, os.path.splitext(network_pkl)[0] + '_' + str(pose_cond))
    os.makedirs(outdir, exist_ok=True)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        G.renderer.plane_axes = G.renderer.plane_axes.to(device)

    pose_cond_rad = pose_cond / 180 * np.pi
    intrinsics = FOV_to_intrinsics(fov_deg, device=device).reshape(-1, 9)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # cond camera settings
        cam_pivot = torch.tensor([0, 0, 0], device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(
            pose_cond_rad, np.pi/2, cam_pivot,
            radius=cam_radius, device=device)
        conditioning_params = torch.cat([
            conditioning_cam2world_pose.reshape(-1, 16),
            intrinsics], 1)
        
        # z and w
        z_np = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z_np).to(device)
        ws = G.mapping(z, conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff)
        G._last_planes = None
        occupancy = G.synthesis_volume(ws)
        verts, faces, normals, values = marching_cubes(occupancy, level=10, spacing=[1, 1, 1])
        verts = torch.from_numpy(verts.copy()).to(device)
        verts = (verts - 256) / 512
        faces = torch.from_numpy(faces.copy()).long().to(device)
        plain_mesh = kal.rep.SurfaceMesh(verts, faces)

        elev = -0.2
        n_views = 8
        azims = np.linspace(-np.pi, np.pi, n_views)
        imgs = []
        cams = []
        for azim in azims:
            # rand camera setting
            cam2world_pose = LookAtPoseSampler.sample(
                np.pi/2 + azim, np.pi/2 + elev, cam_pivot,
                radius=cam_radius, device=device)
            #print(cam2world_pose)
            y = cam_radius * math.sin(elev)
            x = cam_radius * math.sin(azim)
            z = cam_radius * math.cos(azim)
            #print(verts.min(), verts.max(), verts.mean())
            kao_cam = kal.render.camera.Camera.from_args(
                # view_matrix=cam2world_pose,
                eye=torch.tensor([x, y, z]),
                at=torch.tensor([0., 0., 0.]),
                up=torch.tensor([0., 1., 0.]),
                fov=torch.pi * fov_deg / 180,
                width=512, height=512, device=device)
            camera_params = torch.cat([
                cam2world_pose.reshape(-1, 16), intrinsics], 1)

            img = G.synthesis(ws, camera_params, cache_backbone=True)['image']
            img = img.permute(0, 2, 3, 1) * 127.5 + 128
            img = img.clamp(0, 255).to(torch.uint8)
            imgs.append(img)
            cams.append(kao_cam)
        
        for idx, (img, cam) in enumerate(zip(imgs, cams)):
            res_dic = rasterize_mesh(plain_mesh, cam)
            mask = res_dic["fg_mask"][..., None]
            print(mask.sum())
            if mask.sum() < 1:
                continue
            depth = res_dic["depth"]
            normal = res_dic["normal"]
            depth_disp = visualize_depth(mask, depth).repeat(1, 1, 1, 3)
            normal_disp = (normalize(normal) * 255).byte()
            #print(img.device, depth_disp.device, normal_disp.device)
            disp = torch.cat([img, depth_disp, normal_disp], 2)
            imsave(f"test{idx}.png", disp[0].cpu().numpy())

        return occupancy

#----------------------------------------------------------------------------

if __name__ == "__main__":
    occupancy = generate_images() # pylint: disable=no-value-for-parameter
    

#----------------------------------------------------------------------------