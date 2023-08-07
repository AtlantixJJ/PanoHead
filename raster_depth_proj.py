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
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import re
import legacy
import click
import dnnlib
import trimesh
import torch
import numpy as np
import kaolin as kal
import torch.nn.functional as F
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

def rasterize_mesh(mesh, cam: kal.render.camera.Camera, component="depth, normal", intrinsics=None):
    """Rasterize a mesh with a given camera.
    """

    # vertices in camera coordinate
    v_cam = cam.extrinsics.transform(mesh.vertices) # Shape: (1, N, 3)
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
    res_dic = {"fg_mask": face_idx != -1, "face_idx": face_idx}
    fg_mask_img = (np.stack((face_idx[0].cpu().numpy(), face_idx[0].cpu().numpy() ** 2, face_idx[0].cpu().numpy() ** 3), axis=-1) + 1).astype(np.uint8)
    imsave("fg_mask.png", fg_mask_img)
    for idx, name in enumerate(names):
        res_dic[name] = im_features[idx]

    return res_dic
#----------------------------------------------------------------------------

def unproject_depth(depth, cam2world_matrix, intrinsics):
    """
    Convert a depth image DEPTH to points in world space.

    depth: (N, resolution, resolution)
    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    """
    resolution = depth.shape[-1]
    N, M = depth.shape[0], resolution ** 2

    uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
            torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
            indexing='ij')) * (1. / resolution) + (0.5 / resolution)
    uv = uv.reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(N, 1, 1) # shape: (N, M, 2)

    xyz = torch.linalg.solve(intrinsics, torch.concat((uv, 
            torch.ones(N, M, 1).to(intrinsics.device)), dim=-1).permute(0, 2, 1)) 

    cam_rel_points = torch.concat((xyz * depth.view(N, 1, M), torch.ones(N, 1, M).to(intrinsics.device)), dim=1) # shape: (N, 4, M)
    
    world_points = torch.bmm(cam2world_matrix, cam_rel_points).permute(0, 2, 1)[:, :, :3]

    return world_points

#----------------------------------------------------------------------------

def face_area(face_vertices):
    """
    face_vertices: (N, 3, 3)
    RETURNS: (N)
    """
    side_1 = face_vertices[:, 0] - face_vertices[:, 1]
    side_2 = face_vertices[:, 0] - face_vertices[:, 2]
    return torch.linalg.norm(torch.cross(side_1, side_2), dim=-1) / 2

#----------------------------------------------------------------------------

def adaptive_unproject(mesh: kal.rep.SurfaceMesh, n_per_area: int, azims_per_elev: list(),
        elevs: list, cam_pivot, cam_radius, fov_deg, cam_intrinsics, model: TriPlaneGenerator,
        ws, outdir: str):
        assert len(azims_per_elev) == len(elevs)
        areas = face_area(mesh.face_vertices)
        n_per_face = torch.ceil(n_per_area * areas).int()
        points, colors = [], []
        for elev, azims in zip(elevs, azims_per_elev):
            n_per_face_new = torch.zeros(mesh.faces.shape[0], 
                dtype=torch.int32, device=mesh.face_vertices.device)
            for azim in azims:
                cam2world_pose = LookAtPoseSampler.sample2(
                    azim, elev, cam_pivot,
                    radius=cam_radius, device=mesh.face_vertices.device)
                x, y, z = cam2world_pose[0, :3, 3]
                kao_cam = kal.render.camera.Camera.from_args(
                    eye=torch.tensor([x, y, z]),
                    at=torch.tensor([0., 0., 0.]),
                    up=torch.tensor([0., 1., 0.]),
                    fov=torch.pi * fov_deg / 180,
                    width=512, height=512,
                    device=mesh.face_vertices.device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), cam_intrinsics], 1)

                img = model.synthesis(ws, camera_params, cache_backbone=True)['image']
                img = img.permute(0, 2, 3, 1) * 127.5 + 128
                img = img.clamp(0, 255).to(torch.uint8)

                res_dic = rasterize_mesh(mesh, kao_cam, intrinsics=cam_intrinsics)
                mask = torch.logical_and(res_dic["fg_mask"], n_per_face[res_dic["face_idx"]] > 0)
                if mask.sum() < 1:
                    continue

                if np.abs(azim) < np.pi / 40:
                    normal_disp = (normalize(res_dic["normal"]) * 255).byte()
                    disp = torch.cat([img, normal_disp], 2)
                    imsave(f"{outdir}/seed0-({elev:.3f})-({azim:.3f}).png", disp[0].cpu().numpy())

                depth = res_dic["depth"] # (1, 512, 512, 1)
                n_per_face_view = torch.bincount(res_dic["face_idx"].reshape(-1)[mask.view(-1)])
                n_per_face_new[:n_per_face_view.shape[0]] += n_per_face_view
                p = unproject_depth(
                    depth.squeeze(-1), kao_cam.inv_view_matrix(), 
                    cam_intrinsics.reshape(-1, 3, 3))
                points.append(p.reshape(-1, 3)[mask.view(-1)])
                colors.append(img.reshape(-1, 3)[mask.view(-1)])

            n_point_inc = int(n_per_face_new.sum())
            n_per_face = F.relu(n_per_face - n_per_face_new)
            n_point_remain = int(n_per_face.sum())
            print(f"{elev * 12 / np.pi}\t{n_point_inc}\t{n_point_remain}")
            if n_point_remain < 1:
                break
        points = torch.concat(points).cpu().numpy()
        colors = torch.concat(colors).cpu().numpy()
        pcd = trimesh.PointCloud(points, colors)
        pcd.export(f"{outdir}/seed0-adapt.ply")

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
@click.option('--outdir', help='Where to save the output images', type=str, default="out", metavar='DIR')
# @click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=14, show_default=True)
@click.option('--pose_cond', type=int, help='camera conditioned pose angle', default=90, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    fov_deg: float,
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
        conditioning_cam2world_pose = LookAtPoseSampler.sample2(
            np.pi/2 - pose_cond_rad, 0, cam_pivot,
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
        verts, faces, normals, values = marching_cubes(
            occupancy, level=10, spacing=[1, 1, 1])
        verts = torch.from_numpy(verts.copy()).to(device)
        verts = verts / (occupancy.shape[1] - 1) - 0.5
        faces = torch.from_numpy(faces.copy()).long().to(device)
        plain_mesh = kal.rep.SurfaceMesh(verts, faces)

        n_views_1 = 6
        elevs = np.array([0, 1, -1, 2, -2, -2.2, -2.4, -2.6, -2.8, 3]) * np.pi / 12
        #elevs = np.array([0, 1, 2, -1, 3, -2, -2.2, -2.4, -2.6, -2.8]) * np.pi / 12
        azims = [np.linspace(-np.pi, np.pi, n_views_1 + 1)[:n_views_1]] * 6
        azims += [np.linspace(-np.pi / 4, np.pi / 4, 20)] * 4
        adaptive_unproject(plain_mesh, 2000000, azims, elevs.tolist(),
            cam_pivot, cam_radius, fov_deg, intrinsics, G, ws, outdir)

        continue
        n_views = 12
        elevs = [-np.pi / 3] * n_views + [np.pi / 5] * n_views
        azims = np.linspace(-np.pi, np.pi, n_views + 1)[:n_views].tolist() * 2
        colors, points = [], []
        del occupancy, verts, faces
        for idx, (elev, azim) in enumerate(zip(elevs, azims)):
            # rand camera setting
            cam2world_pose = LookAtPoseSampler.sample2(
                azim, elev, cam_pivot,
                radius=cam_radius, device=device)
            #print(cam2world_pose)
            x, y, z = cam2world_pose[0, :3, 3]
            kao_cam = kal.render.camera.Camera.from_args(
                eye=torch.tensor([x, y, z]),
                at=torch.tensor([0., 0., 0.]),
                up=torch.tensor([0., 1., 0.]),
                fov=torch.pi * fov_deg / 180,
                width=512, height=512,
                device=device)
            camera_params = torch.cat([
                cam2world_pose.reshape(-1, 16), intrinsics], 1)

            img = G.synthesis(ws, camera_params, cache_backbone=True)['image']
            img = img.permute(0, 2, 3, 1) * 127.5 + 128
            img = img.clamp(0, 255).to(torch.uint8)

            res_dic = rasterize_mesh(plain_mesh, kao_cam, intrinsics=intrinsics)
            mask = res_dic["fg_mask"][..., None]
            print(mask.sum())
        
            if mask.sum() < 1:
                continue
            depth = res_dic["depth"]
            normal = res_dic["normal"]
            points.append(unproject_depth(depth.squeeze(-1), kao_cam.inv_view_matrix(), 
                            intrinsics.reshape(-1, 3, 3)).reshape(-1, 3)[mask.view(-1)])
            colors.append(img.reshape(-1, 3)[mask.view(-1)])
            # depth_disp = visualize_depth(mask, depth).repeat(1, 1, 1, 3)
            # normal_disp = (normalize(normal) * 255).byte()
            #print(img.device, depth_disp.device, normal_disp.device)
            # disp = torch.cat([img, depth_disp, normal_disp], 2)
            # imsave(f"{outdir}/test{idx}.png", disp[0].cpu().numpy())
            pcd = trimesh.PointCloud(points[-1].cpu().numpy(), colors[-1].cpu().numpy())
            # pcd.export(f"{outdir}/seed{seed}-test{idx}.ply")

        points = torch.concat(points).cpu().numpy()
        colors = torch.concat(colors).cpu().numpy()
        pcd = trimesh.PointCloud(points, colors)
        pcd.export(f"{outdir}/seed{seed}.ply")
        # return occupancy


if __name__ == "__main__":
    occupancy = generate_images() # pylint: disable=no-value-for-parameter
    

#----------------------------------------------------------------------------
