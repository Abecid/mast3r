import time
import argparse

import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for feature matching.")
    parser.add_argument('--model_name', type=str, required=False, help='Name or path of the pretrained model')
    parser.add_argument('--capture_path', type=str, required=False, help='Path to the directory containing the captured images')
    args = parser.parse_args()

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    dir_path = "/home/fai/workspace/104/vco_vol/log/capture/2025_01_13_11_27_38_872" if args.capture_path is None else args.capture_path
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric" if args.model_name is None else args.model_name
    # you can put the path to a local checkpoint in model_name if needed
    start_time = time.time()
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    print ("load : ", time.time() - start_time)

    capture_str = dir_path.split("/")[-1]
    images_paths = [f"{dir_path}/{capture_str}_LW_L.png", f"{dir_path}/{capture_str}_TL_L.png"]

    images = load_images(images_paths, size=512)

    start_time = time.time()
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    print ("inference : ", time.time() - start_time)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # Visualize selected matches
    n_viz = 2000
    num_matches = matches_im0.shape[0]
    print("num_matches : ", num_matches)
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

    # Save the visualization
    output_path = "output/matches.jpg"
    pl.savefig(output_path, format='jpg', dpi=300)  # Save the figure with high resolution
    print(f"Saved matches visualization to {output_path}")

    pl.show(block=True)