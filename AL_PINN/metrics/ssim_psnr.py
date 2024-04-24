import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
from tqdm import tqdm
import re


def calculate_ssim(img_1, img_2):
    """img_1 and img_2 npy files with shape (32, 2, 100, 300)"""
    ssim_sum = 0
    # align dimension
    if len(img_1.shape) == 3:
        img_1 = np.expand_dims(img_1, axis=0)
    if len(img_2.shape) == 3:
        img_2 = np.expand_dims(img_2, axis=0)

    for i in range(img_1.shape[0]):
        ssim_sum += ssim(img_1[i, :, :, :].transpose(1, 2, 0),
                         img_2[i, :, :, :].transpose(1, 2, 0),
                         multichannel=True)
    return ssim_sum / img_1.shape[0]


def calculate_psnr(img_1, img_2):
    """Calculate PSNR between two images."""
    psnr_sum = 0
    # align dimension
    if len(img_1.shape) == 3:
        img_1 = np.expand_dims(img_1, axis=0)
    if len(img_2.shape) == 3:
        img_2 = np.expand_dims(img_2, axis=0)
        
    for i in range(img_1.shape[0]):
        psnr_sum += psnr(img_1[i, :, :, :].transpose(1, 2, 0),
                         img_2[i, :, :, :].transpose(1, 2, 0), data_range=255)
    return psnr_sum / img_1.shape[0]


def get_file_name(path):
    return re.search(r"cylinder_(\d{1,3}|1000)", path).group(0) if 'cylinder' in path else re.search(r"airfoil_(\d{4})_(\d{1})", path).group(0)


def dataset_avg_info(dataset1: str, dataset2: str, func: callable, axis_for_calculation = 0) -> float:
    """
    Compares two datasets of images and calculates statistics (mean, max, min, standard deviation) 
    of the comparison results.
    # Select the axis for the calculation based on the desired shape (x*n or n*x), x is the number of return value, n is the number of samples
    # For shape (x, n): axis=1, for shape (n, x): axis=0
    # Assuming the desired shape is (n, x) for this example

    Parameters:
    - dataset1: Path to the directory containing the ground truth images.
    - dataset2: Path to the directory containing the generated images.
    - func: A callable that takes two numpy arrays as input (representing two images) and returns a tuple of floats.
    """
    data_ls = []
    for f in os.listdir(dataset1):
        fname = os.path.splitext(f)[0]  # Assumes filename without extension
        if not os.path.exists(os.path.join(dataset2, f'{fname}.npy')):
            if os.path.exists(os.path.join(dataset2, f'{fname}_result.npy')):
                fname = f'{fname}_result'
            elif os.path.exists(os.path.join(dataset2, f'{fname}_v.npy')):
                fname = f'{fname}_v'
            else:
                print(f'{fname} does not exist in {dataset2}')
                return None
            
        image2_path = os.path.join(dataset2, f'{fname}_result.npy') if os.path.exists(
            os.path.join(dataset2, f'{fname}_result.npy')) else os.path.join(dataset2, f'{fname}.npy')
        if os.path.exists(image2_path):
            image1 = np.load(os.path.join(dataset1, f))
            image2 = np.load(image2_path)
            data_ls.append(np.array(func(image1, image2)))

    # Ensure that data_ls is not empty
    if not data_ls:
        print("No comparable image pairs found.")
        return 0

    stacked_data = np.stack(data_ls)

    avg = np.mean(stacked_data, axis=axis_for_calculation)
    max_val = np.max(stacked_data, axis=axis_for_calculation)
    min_val = np.min(stacked_data, axis=axis_for_calculation)
    std_dev = np.std(stacked_data, axis=axis_for_calculation)

    print(f"{func.__name__} - Average: {avg}, Max: {max_val}, Min: {min_val}, Std Dev: {std_dev}, Number of images: {len(data_ls)}")
    
    print(stacked_data.shape)
    return stacked_data


def dataset_avg_p(dataset1: str, dataset2: str, func: callable, axis_for_calculation = 0) -> float:
    """
    Compares two datasets of images and calculates statistics (mean, max, min, standard deviation) 
    of the comparison results.
    # Select the axis for the calculation based on the desired shape (x*n or n*x), x is the number of return value, n is the number of samples
    # For shape (x, n): axis=1, for shape (n, x): axis=0
    # Assuming the desired shape is (n, x) for this example

    Parameters:
    - dataset1: Path to the directory containing the ground truth images.
    - dataset2: Path to the directory containing the generated images.
    - func: A callable that takes two numpy arrays as input (representing two images) and returns a tuple of floats.
    """
    data_ls = []
    for f in os.listdir(dataset1):
        fname = os.path.splitext(f)[0]  # Assumes filename without extension
        if not os.path.exists(os.path.join(dataset2, f'{fname}.npy')):
            if os.path.exists(os.path.join(dataset2, f'{fname}_result.npy')):
                fname = f'{fname}_result'
            elif os.path.exists(os.path.join(dataset2, f'{fname}_v.npy')):
                fname = f'{fname}_v'

        image2_path = os.path.join(dataset2, f'{fname}.npy')
        if os.path.exists(image2_path):
            image1 = np.load(os.path.join(dataset1, f))[:, 0, :, :]
            image2 = np.load(image2_path)[:, 0, :, :]
            data_ls.append(np.array(func(image1, image2)))

    # Ensure that data_ls is not empty
    if not data_ls:
        print("No comparable image pairs found.")
        return 0

    stacked_data = np.stack(data_ls)

    avg = np.mean(stacked_data, axis=axis_for_calculation)
    max_val = np.max(stacked_data, axis=axis_for_calculation)
    min_val = np.min(stacked_data, axis=axis_for_calculation)
    std_dev = np.std(stacked_data, axis=axis_for_calculation)

    print(f"{func.__name__} - Average: {avg}, Max: {max_val}, Min: {min_val}, Std Dev: {std_dev}, Number of images: {len(data_ls)}")
    
    print(stacked_data.shape)
    return stacked_data
