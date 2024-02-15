# evaluate_dycheck.py
# Description: This script is used to evaluate the performance of the model on the dycheck dataset
#               by computing mPSNR and mSSIM given path to two sets of images (original and rendered)
# Inputs:
#  - img_dir: path to the directory containing the rendered images
#  - gt_dir: path to the directory containing the ground truth images
#  - mask_dir: path to the directory of the scene, should be /path/to/dataset
#              where rgb, depth, covisible, dataset.json, etc are under it.

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm


def compare_psnr(gt, img, mask):
    eps = 1e-6
    mse = (gt - img) ** 2
    mask = np.broadcast_to(mask[..., None], mse.shape)
    masked_mse = (mse * mask).sum() / np.clip(mask.sum(), eps, None) 
    return -10.0 * np.log10(10.0) * np.log10(masked_mse)

def compare_ssim(gt, img):
    return structural_similarity(gt, img, win_size=11, channel_axis=2, data_range=1.0)


def eprint(*args):
    print(*args, file=sys.stderr)


def report_metrics(gt_dir, img_dir, dataset_dir, out_dir, metrics, img_str="%05d.png", gt_str="%05d.png", use_gpu=False, print_info=True, prefix="", output_video=None):
    if output_video is not None and not isinstance(output_video, str):
        raise ValueError("output_video should be a path")

    total ={}
    loss_fn, loss_fn_vgg = None, None

    if print_info:
        eprint(gt_dir, img_dir, out_dir)
        eprint(img_str, gt_str)

    # * Parse dataset
    val_ids = []
    dataset_json_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
        for _id in dataset_json["val_ids"]:
            val_ids.append(_id)
    
    mask_path_list = [os.path.join(dataset_dir, "covisible", "2x", "val", f"{_id}.png") for _id in val_ids]

    n_images = 0
    while True:
        img = os.path.isfile(os.path.join(img_dir, img_str % n_images))
        gt = os.path.isfile(os.path.join(gt_dir, gt_str % n_images))
        if not (img and gt):
            break
        n_images += 1
    print("Total images: ", n_images)

    assert n_images == len(val_ids), "Number of images in the dataset and the number of images in the directories do not match"

    combined_photos = []
    
    for i in tqdm(range(n_images)):
        img = cv2.imread(os.path.join(img_dir, img_str % i)) # (H, W, 3)
        gt = cv2.imread(os.path.join(gt_dir, gt_str % i)) # (H, W, 3)
        mask = np.array(Image.open(mask_path_list[i])) # (H, W)

        # Resize to same scale
        if img.shape[0] != mask.shape[0]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        if gt.shape[0] != mask.shape[0]:
            gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]))

        img = np.asarray(img, np.float32) / 255.0
        gt = np.asarray(gt, np.float32) / 255.0

        if output_video:
            combined_img = cv2.hconcat([img, gt])
            combined_photos.append((combined_img * 255).astype(np.uint8))
        if i == 0 and print_info:
            eprint("img.shape", img.shape)
            eprint("gt.shape", gt.shape)
            eprint("mask.shape", mask.shape)

        for key in metrics:
            key = key.lower()
            if key == "psnr":
                val = compare_psnr(gt, img, mask)
            elif key == "ssim":
                val = compare_ssim(gt, img)
            else:
                raise NotImplementedError("metrics of {} not implemented".format(key))
            if key not in total:
                total[key] = [val]
            else:
                total[key].append(val)

    if output_video:
        height, width, _ = combined_photos[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(os.path.join(args.output_video), fourcc, 2, (width, height))

        for frame in combined_photos:
            output_video.write(frame)

        output_video.release()

    del loss_fn
    del loss_fn_vgg

    torch.cuda.empty_cache()
    if len(total) > 0:
        outStr = []
        for key in total.keys():
            vals = np.asarray(total[key]).reshape(-1)
            np.savetxt(os.path.join(out_dir, f'{key}.txt'), vals)
            outStr.append("%.6f" % np.mean(vals))

        outStr = ",".join(outStr)
        if prefix:
            outStr = prefix + "," + outStr

        print(outStr)
        with open(os.path.join(out_dir, "scores.txt"), "w+") as f:
            print(outStr, file=f)


def parse_args():
    parser = argparse.ArgumentParser(description="compute scores")

    parser.add_argument('-i', '--img_dir', help="The folder that contain output images.")
    parser.add_argument('-g', '--gt_dir', default=None, help="The folder that contain gt images. By default it uses img_dir")
    parser.add_argument('-o', '--out_dir', default=None, help="The folder that contain output files. By default it uses img_dir")
    parser.add_argument('-d', '--dataset_dir', help="The folder that contain the dataset.")
    parser.add_argument('-is', '--img_str', default="%05d.png", help="The string format for input images.")
    parser.add_argument('-gs', '--gt_str', default="%05d.png", help="The string format for GT images.")
    parser.add_argument('-m', '--metrics', nargs='+', default=["psnr", "ssim", "lpips", "vgglpips"],  help="The list of metrics to compute. By default it computes psnr, ssim and rmse.")
    parser.add_argument('--prefix', default="")
    parser.add_argument('--output_video', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.gt_dir is None:
        args.gt_dir = args.img_dir
    if args.out_dir is None:
        args.out_dir = args.img_dir
    if args.dataset_dir is None:
        raise ValueError("Please provide the dataset for covisible mask to get correct mPSNR and mSSIM.")

    report_metrics(args.gt_dir, args.img_dir, args.dataset_dir, args.out_dir, args.metrics, img_str=args.img_str, gt_str=args.gt_str, use_gpu=True, print_info=args.verbose, prefix=args.prefix, output_video=args.output_video)