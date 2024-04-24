from ssim_psnr import calculate_ssim, calculate_psnr, dataset_avg_p, dataset_avg_info
import os
import os.path as osp
from evaluate_freq_and_L1_diff import freq_and_L1_diff
import pandas as pd
import argparse
from typing import Callable


def inference(gt_dir: str, dataset_dir: str, feature: str, func: Callable) -> None:
    print(feature)
    ssim_val = func(gt_dir, dataset_dir, calculate_ssim)
    psnr_val = func(gt_dir, dataset_dir, calculate_psnr)
    return ssim_val, psnr_val


def get_name(path: str):
    '''input file path. return (fname, fext).'''
    return osp.splitext(osp.basename(osp.normpath(path)))


def save_excel(args):
    print('=====================dt4=====================')
    l1 = dataset_avg_info(args.gt_dir, args.ds_dir, freq_and_L1_diff)
    if l1 is None:
        print("Error in calculating L1.")
        return
    results = {}
    prefix = get_name(args.ds_dir)[0]

    results[f'{prefix}_l1_freq_p'] = l1[:, 0]  # assuming l1 returns a tuple of four values
    results[f'{prefix}_l1_freq_a'] = l1[:, 1]
    results[f'{prefix}_l1_p'] = l1[:, 2]
    results[f'{prefix}_l1_a'] = l1[:, 3]

    ds_func = dataset_avg_p if not args.use_v else dataset_avg_info
    ssim, psnr = inference(args.gt_dir, args.ds_dir, "placeholder", ds_func)
    results[f'{prefix}_ssim'] = ssim
    results[f'{prefix}_psnr'] = psnr

    df = pd.DataFrame(results)
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    df.to_excel(os.path.join(args.store_dir, f"{prefix}_{args.feature}{'_v' if args.use_v else ''}.xlsx"), index=False)

    print("Results saved to Excel.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='ground_truth')
    parser.add_argument('--ds_dir', type=str, default='test_dataset')
    parser.add_argument('--feature', type=str, default='bl_v')
    parser.add_argument('--use_v', action='store_true')
    parser.add_argument('--store_dir', type=str, default='/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res')

    args = parser.parse_args()

    save_excel(args)