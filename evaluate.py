import argparse
import os
import sys
import time

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm


def compare_ssim(gt, img):
    return structural_similarity(gt, img, win_size=11, channel_axis=2, data_range=1.0)


def eprint(*args):
    print(*args, file=sys.stderr)


def report_metrics(gtFolder, imgFolder, outFolder, metrics, imgStr="%05d.png", gtStr="%05d.png", use_gpu=False, print_info=True, prefix="", output_video=None):
    if output_video is not None and not isinstance(output_video, str):
        raise ValueError("output_video should be a path")

    total ={}
    loss_fn, loss_fn_vgg = None, None

    if print_info:
        eprint(gtFolder, imgFolder, outFolder)
        eprint(imgStr, gtStr)
    if "lpips" in metrics:
        loss_fn = lpips.LPIPS(net='alex', version='0.1', verbose=False) 
        loss_fn = loss_fn.cuda() if use_gpu else loss_fn
    if "vgglpips" in metrics:
        loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1', verbose=False)
        loss_fn_vgg = loss_fn_vgg.cuda() if use_gpu else loss_fn_vgg

    combined_photos = []

    n_images = 0
    while True:
        img = os.path.isfile(os.path.join(imgFolder, imgStr % n_images))
        gt = os.path.isfile(os.path.join(gtFolder, gtStr % n_images))
        if not (img and gt):
            break
        n_images += 1
    print("Total images: ", n_images)

    combined_photos = []
    
    for i in tqdm(range(n_images)):
        img = cv2.imread(os.path.join(imgFolder, imgStr % i))
        gt = cv2.imread(os.path.join(gtFolder, gtStr % i))

        img = np.asarray(img, np.float32) / 255.0
        gt = np.asarray(gt, np.float32) / 255.0

        if output_video:
            combined_img = cv2.hconcat([img, gt])
            combined_photos.append((combined_img * 255).astype(np.uint8))

        for key in metrics:
            key = key.lower()
            if key == "psnr":
                val = compare_psnr(gt, img)
            elif key == "ssim":
                val = compare_ssim(gt, img)
            elif key == "lpips":
                # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                gt_tensor = torch.from_numpy(gt)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                img_tensor = img_tensor.cuda() if use_gpu else img_tensor
                gt_tensor = gt_tensor.cuda() if use_gpu else gt_tensor
                val = loss_fn(img_tensor, gt_tensor).item()
            elif key == "vgglpips":
                # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                gt_tensor = torch.from_numpy(gt)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                img_tensor = img_tensor.cuda() if use_gpu else img_tensor
                gt_tensor = gt_tensor.cuda() if use_gpu else gt_tensor
                val = loss_fn_vgg(img_tensor, gt_tensor).item()
            elif key == "rmse":
                val = np.sqrt(mean_squared_error(gt, img))
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
            np.savetxt(os.path.join(outFolder, f'{key}.txt'), vals)
            outStr.append("%.6f" % np.mean(vals))

        outStr = ",".join(outStr)
        if prefix:
            outStr = prefix + "," + outStr

        print(outStr)
        with open(os.path.join(outFolder, "scores.txt"), "w+") as f:
            print(outStr, file=f)


def parse_args():
    parser = argparse.ArgumentParser(description="compute scores")

    parser.add_argument('-i', '--imgFolder', help="The folder that contain output images.")
    parser.add_argument('-g', '--gtFolder', default=None, help="The folder that contain gt images. By default it uses imgFolder")
    parser.add_argument('-o', '--outFolder', default=None, help="The folder that contain output files. By default it uses imgFolder")
    parser.add_argument('-is', '--imgStr', default="%05d.png", help="The string format for input images.")
    parser.add_argument('-gs', '--gtStr', default="%05d.png", help="The string format for GT images.")
    parser.add_argument('-m', '--metrics', nargs='+', default=["psnr", "ssim", "lpips", "vgglpips"],  help="The list of metrics to compute. By default it computes psnr, ssim and rmse.")
    parser.add_argument('--prefix', default="")
    parser.add_argument('--output_video', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.gtFolder is None:
        args.gtFolder = args.imgFolder
    if args.outFolder is None:
        args.outFolder = args.imgFolder

    report_metrics(args.gtFolder, args.imgFolder, args.outFolder, args.metrics, imgStr=args.imgStr, gtStr=args.gtStr, use_gpu=True, print_info=args.verbose, prefix=args.prefix, output_video=args.output_video)



