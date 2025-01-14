import os
import numpy as np

import torch
import trimesh
from scipy.spatial.transform import Rotation

import tempfile
import shutil
from contextlib import nullcontext

from mast3r.demo import get_args_parser, main_demo

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess


import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl
import functools

import argparse
import open3d as o3d

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=640, height=480, visible=True)
vis.get_render_option().point_size = 5
pcd_mkpts3d = o3d.geometry.PointCloud()

def _center_view(vis):
    vis.reset_view_point(True)


vis.register_key_callback(ord("C"), _center_view)



class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--confidence_threshold", type=float, default=3.0,
                        help="confidence values higher than threshold are invalid")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--pnp_mode", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error", type=float, default=5.0, help="pnp reprojection error")
    parser_reproj.add_argument("--reprojection_error_diag_ratio", type=float, default=None,
                               help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches", type=int, default=0, help="debug matches")

    parser.add_argument("--image_size", type=int, default=512, help="512 (default) or 224")

    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    print (np.array(pts3d).shape)
    #pcd_all = []
    # for pts in pts3d:
    #     pcd_all.append(pts)

    # print (np.array(pcd_all).shape)
    for i in range(len(pts3d)):
        pcd_mkpts3d.points = o3d.utility.Vector3dVector(np.array(pts3d[i]))
    vis.add_geometry(pcd_mkpts3d)
    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj="./temp.glb")
    
    return outfile


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)



def recon(filelist, image_size):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)

    """
    # lr1, niter1 : learning rate and #iterations for coarse global alignment (3D matching)
    # lr2, niter2: learning rate and #iterations for refinement (2D reproj error)
    # scenegraph_type : complete: all possible image pairs

    lr1 = 0.07
    niter1 = 500
    lr2 = 0.014
    niter2 = 200
    matching_conf_thr = 5.
    shared_intrinsics = False
    winsize = 1
    win_cyclic = False
    optim_level = "refine+depth"
    scenegraph_type = "complete"

    min_conf_thr = 1.5
    cam_size = 0.2
    TSDF_thresh = 0.
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    device = args.device
    cache_dir = "./cache"

    silent = False

    scene_graph_params = [scenegraph_type]

    scene_graph = '-'.join(scene_graph_params)

    imgs = load_images(filelist, size=image_size)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    if optim_level == 'coarse':
        niter2 = 0

    
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
            model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
            opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr)
    
    outfile_name = 'scene.glb'
    scene_state = SparseGAState(scene, 1, cache_dir, outfile_name)

    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    print (outfile)

    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])


    print ("rgbimg ", rgbimg[0].shape, len(rgbimg))
    _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


if __name__ == '__main__':
    """
    python demo_recon.py --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    """
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    weights_path = args.weights

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    filelist = [
        "/home/jhna/git/accelerated_features/samples/images/LW_R3.png", 
        "/home/jhna/git/accelerated_features/samples/images/TB_L5.png", 
        "/home/jhna/git/accelerated_features/samples/images/TB_R5.png"
    ]


    #import glob
    #source_path = "/home/jhna/git/accelerated_features/samples/images"
    #filelist = [f for f in glob.glob(source_path + '/*_L*.png')]
    
    recon(filelist, args.image_size)
    while True:
        vis.poll_events()
        vis.update_renderer()
