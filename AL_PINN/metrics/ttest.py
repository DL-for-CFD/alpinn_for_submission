import numpy as np
import os
from scipy.stats import ttest_ind
import re
from ssim_psnr import calculate_ssim, calculate_psnr


def dataset_comparison_stats(dataset1: str, dataset2: str, func: callable) -> float:
    """
    Compares two datasets of images, calculates statistics, and performs a t-test on the results.
    Parameters:
    - dataset1: Path to the directory containing images from method 1.
    - dataset2: Path to the directory containing images from method 2.
    - func: A callable that takes two numpy arrays as input (representing two images).
    """
    results1 = []
    results2 = []
    
    for f in os.listdir(dataset1):
        img1_path = os.path.join(dataset1, f)
        img2_path = os.path.join(dataset2, f)
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = np.load(img1_path)
            img2 = np.load(img2_path)
            results1.append(func(img1, img2))
            results2.append(func(img2, img1))  # Assuming you want to reverse compare as well

    if not results1 or not results2:
        print("No comparable image pairs found in one or both datasets.")
        return None

    results1 = np.array(results1)
    results2 = np.array(results2)
    
    avg1, max1, min1, std_dev1 = results1.mean(), results1.max(), results1.min(), results1.std()
    avg2, max2, min2, std_dev2 = results2.mean(), results2.max(), results2.min(), results2.std()
    
    t_stat, p_value = ttest_ind(results1, results2)

    print(f"Dataset 1 - {func.__name__} - Average: {avg1}, Max: {max1}, Min: {min1}, Std Dev: {std_dev1}")
    print(f"Dataset 2 - {func.__name__} - Average: {avg2}, Max: {max2}, Min: {min2}, Std Dev: {std_dev2}")
    print(f"T-test results: T-statistic = {t_stat}, P-value = {p_value}")
    
    return avg1, avg2, p_value

if __name__ == "__main__":
    print('=====================dt4=====================')
    root: str = '/csproject/t3_lzengaf/lzengaf/fyp'
    overfitted = '/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted'
    auto_airfoil = '/project/t3_zxiaoal/Validation_Dataset/dt4'
    baseline = '/project/t3_zxiaoal/Validation_Dataset/dt4_baseline'

    # avg1, avg2, p_value = dataset_comparison_stats('path_to_dataset1', 'path_to_dataset2', calculate_ssim)
    # avg1, avg2, p_value = dataset_comparison_stats('path_to_dataset1', 'path_to_dataset2', calculate_psnr)
