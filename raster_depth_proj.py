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
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def rasterize_mesh(mesh, cam: kal.render.camera.Camera, component="depth,normal", intrinsics=None):
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

def adaptive_unproject(
        mesh: kal.rep.SurfaceMesh,
        n_per_area: int,
        azims_per_elev: list(),
        elevs: list,
        cam_pivot,
        cam_radius,
        fov_deg,
        cam_intrinsics,
        model: TriPlaneGenerator,
        ws,
        outdir: str):
    assert len(azims_per_elev) == len(elevs)

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
            mask = res_dic["fg_mask"] & (n_per_face[res_dic["face_idx"]] > 0)
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


def kaolin_camera_euler(elev, azim, radius, look_at, fov_deg,
                        resolution=512, device="cuda"):
    """Kaolin camera looking at a position.
    Args:
        elev: elevation angle in radian.
        azim: azimus angle in radian.
        look_at: The look at position of the camera.
        radius: the distance of camera to look_at.
        fov_deg: Field of View in degree.
    Returns:
        Kaolin camera object, camera to world transformation matrix
    """
    cam2world = LookAtPoseSampler.sample2(
        azim, elev, look_at,
        radius=radius, device=device)
    kao_cam = kal.render.camera.Camera.from_args(
        eye=cam2world[0, :3, 3],
        at=torch.tensor([0., 0., 0.]),
        up=torch.tensor([0., 1., 0.]),
        fov=torch.pi * fov_deg / 180,
        width=resolution, height=resolution,
        device=device)
    return kao_cam, cam2world


def laplace_image(img):
    diff = torch.zeros_like(img)
    diff[:, 1:] = (img[:, 1:, :] - img[:, :-1, :]).abs()
    diff[:, :, 1:] += (img[:, :, 1:] - img[:, :, :-1]).abs()
    return diff


def arange_abs(n):
    ind = torch.arange(n)
    return torch.stack([ind, -ind], -1).view(-1)[1:]


def n_azim_given_elev(elev):
    """Return different azimus given elevation.
    Higher elevation results in smaller number of azimus division.
    """
    if abs(elev) > torch.pi * 3 / 8:
        return 4
    if abs(elev) > torch.pi * 2 / 8:
        return 6
    if abs(elev) > torch.pi * 1 / 8:
        return 8
    return 10

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
@click.option('--max_ppa', 'max_ppa', type=float, help='Maximum number of points per area (512^3 volume grid).', default=1, show_default=True)
# @click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=14, show_default=True)
@click.option('--pose_cond', type=int, help='camera conditioned pose angle', default=90, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    max_ppa: int,
    fov_deg: float,
    pose_cond: int,
):
    """Generate images using pretrained network pickle.
    """
    torch.set_grad_enabled(False)

    print('=> Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    network_pkl = os.path.basename(network_pkl)
    os.makedirs(outdir, exist_ok=True)

    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new
    G.renderer.plane_axes = G.renderer.plane_axes.to(device)

    pose_cond_rad = pose_cond / 180 * np.pi
    intrinsics = FOV_to_intrinsics(fov_deg, device=device).reshape(-1, 9)
    look_at = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # cond camera settings
        conditioning_cam2world_pose = LookAtPoseSampler.sample2(
            np.pi / 2 - pose_cond_rad, 0, look_at,
            radius=cam_radius, device=device)
        conditioning_params = torch.cat([
            conditioning_cam2world_pose.reshape(-1, 16),
            intrinsics], 1)

        # z and w
        z_np = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z_np).float().to(device)
        ws = G.mapping(z, conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff)
        G._last_planes = None # reset the cache for a new w.
        occupancy = G.synthesis_volume(ws)
        verts, faces, normals, values = marching_cubes(
            occupancy, level=10, spacing=[1, 1, 1])
        verts = torch.from_numpy(verts.copy()).to(device)
        # convert vertices to [-0.5, 0.5] volume
        verts = verts / (occupancy.shape[1] - 1) - 0.5
        faces = torch.from_numpy(faces.copy()).long().to(device)
        plain_mesh = kal.rep.SurfaceMesh(verts, faces)
        scaled_vertices = (plain_mesh.face_vertices + 0.5) * (occupancy.shape[1] - 1)
        areas = face_area(scaled_vertices)
        n_appface = torch.ceil(max_ppa * areas).int()
        max_appface = n_appface.clone()
        N_FACE = n_appface.shape[0]
        K = 3
        dilate_kernel = torch.ones((1, 1, K, K)).to(z)

        N_ELEV = 6
        elevs = arange_abs(N_ELEV // 2).float()
        # elev=0 and azim=0/90/180/270 produces artifacts in rendering
        elevs[0] += 1e-3 
        elevs = torch.Tensor(elevs) * torch.pi / N_ELEV
        angles = []
        for elev in elevs:
            n_azim = n_azim_given_elev(elev)
            azims = arange_abs(n_azim) * torch.pi * 2 / n_azim
            angles.extend([[elev, azim] for azim in azims])
        angles = torch.Tensor(angles)
        face_points, face_colors = [], []
        ratio_fullface = []
        ratio_emptyface = []
        for idx, (elev, azim) in enumerate(tqdm(angles)):
            kao_cam, cam2world = kaolin_camera_euler(
                elev, azim, cam_radius, look_at, fov_deg)
            camera_params = torch.cat([
                cam2world.reshape(-1, 16), intrinsics], 1)

            img = G.synthesis(ws, camera_params, cache_backbone=True)['image']
            img = img.permute(0, 2, 3, 1) * 127.5 + 128
            img = img.clamp(0, 255).to(torch.uint8)

            res_dic = rasterize_mesh(plain_mesh, kao_cam, intrinsics=intrinsics)

            # filter by the remaining number of points per face
            mask = res_dic["fg_mask"] & (n_appface[res_dic["face_idx"]] > 0)
            
            # filter by the normal with camera
            cam_center = kao_cam.inv_view_matrix()[0, :3, -1]
            cam_center = cam_center / (1e-8 + cam_center.norm())
            dot = (res_dic["normal"] @ cam_center)
            edge_image = (dot > -0.5).float()[:, None]
            edge_mask = F.conv2d(edge_image, dilate_kernel, padding=K // 2)[:, 0, ..., None] < 0.9
            #plt.hist(dot[res_dic["fg_mask"]].cpu().view(-1).numpy(), bins=100)
            #plt.savefig(f"{outdir}/hist_{idx}.png")
            #plt.close()
            mask &= edge_mask[..., 0]

            ## filter by the laplace in depth
            ## see if depth can be contained by dilating normal filtering
            #delta_depth = laplace_image(res_dic["depth"]) # (1, 512, 512, 1)
            #plt.hist(delta_depth[res_dic["fg_mask"]].cpu().view(-1).numpy(), #bins=100)
            #plt.savefig(f"{outdir}/hist_{idx}.png")
            #plt.close()
            # dilate the edge
            #edge_image = (delta_depth[:, None, ..., 0] > 0.02).float()
            #delta_depth_mask = F.conv2d(edge_image, dilate_kernel, padding=K // 2)[:, 0, ..., None] < 0.9
            #mask &= delta_depth_mask[..., 0]

            fg_mask_disp = (res_dic["fg_mask"].byte() * 255)[..., None].repeat(1, 1, 1, 3)
            normal_disp = (normalize(res_dic["normal"]) * 255).byte()
            depth_disp = (visualize_depth(res_dic["fg_mask"][..., None], res_dic["depth"])).repeat(1, 1, 1, 3)
            edge_disp = (edge_mask.byte() * 255).repeat(1, 1, 1, 3)
            disp = torch.cat([img, fg_mask_disp, normal_disp, depth_disp, edge_disp], 2)
            imsave(f"{outdir}/seed{seed_idx}_viz{idx}_{elev:.3f}_{azim:.3f}.png", disp[0].cpu().numpy())

            if mask.sum() < 1:
                continue

            depth = res_dic["depth"] # (1, 512, 512, 1)
            n_ppface = torch.bincount(
                res_dic["face_idx"].reshape(-1)[mask.view(-1)])
            max_idx = n_ppface.shape[0]
            n_appface[:max_idx] = F.relu(n_appface[:max_idx] - n_ppface)
            points = unproject_depth(
                depth.squeeze(-1), kao_cam.inv_view_matrix(), 
                intrinsics.reshape(-1, 3, 3))
            points = points.reshape(-1, 3)[mask.view(-1)].cpu().numpy()
            colors = img.reshape(-1, 3)[mask.view(-1)].cpu().numpy()
            face_points.append(points)
            face_colors.append(colors)
            trimesh.PointCloud(points, colors).export(f"{outdir}/seed{seed_idx}_{idx}_{elev:.3f}_{azim:.3f}.ply")
            trimesh.PointCloud(np.concatenate(face_points), np.concatenate(face_colors)).export(f"{outdir}/seed{seed_idx}_acc{idx}_{elev:.3f}_{azim:.3f}.ply")

            n_emptyface = (max_appface == n_appface).float().sum()
            n_fullface = (n_appface == 0).float().sum()
            ratio_emptyface.append(float(n_emptyface / N_FACE))
            ratio_fullface.append(float(n_fullface / N_FACE))
        face_points = np.concatenate(face_points)
        face_colors = np.concatenate(face_colors)
        trimesh.PointCloud(face_points, face_colors).export(
            f"{outdir}/seed{seed_idx}.ply")
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(1, 2, 1)
        ax.plot(ratio_emptyface)
        ax.set_title("Ratio of Empty Triangles")
        ax = plt.subplot(1, 2, 2)
        ax.plot(ratio_fullface)
        ax.set_title("Ratio of Full Triangles")
        plt.savefig(f"{outdir}/seed{seed_idx}_ratio.png")
        plt.close()

if __name__ == "__main__":
    occupancy = generate_images() # pylint: disable=no-value-for-parameter
    

#----------------------------------------------------------------------------
