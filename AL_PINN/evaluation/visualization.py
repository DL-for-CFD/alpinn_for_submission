import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F


def get_name(path:str):
    '''input file path. return (fname, fext).'''
    return osp.splitext(osp.basename(osp.normpath(path)))


def toCuda(x):
    if type(x) is tuple:
        return [xi.cuda() if torch.cuda.is_available() else xi for xi in x]
    return x.cuda() if torch.cuda.is_available() else x


def rot_mac(a):
    return torch.cat([-dx_right(a), dy_bottom(a)], dim=1)


dx_right_kernel = toCuda(
    torch.Tensor([0, -1, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))


def dx_right(v):
    return F.conv2d(v, dx_right_kernel, padding=(0, 1))


dy_bottom_kernel = toCuda(
    torch.Tensor([0, -1, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))


def dy_bottom(v):
    return F.conv2d(v, dy_bottom_kernel, padding=(1, 0))


def get_v(data):
    a = data[:, 0, :, :]
    v = []
    # iterate ofver data[:]
    for i in range(0, data.shape[0]):
        v.append(rot_mac(a[i]))
    return torch.stack(v)


def plot_gif(data_dir, store_dir, num_frames=32):
    for f in tqdm(os.listdir(data_dir)):
        if f.endswith('.npy'):
            fname, fext = get_name(f)
            data = np.load(os.path.join(data_dir, f))
            data = data[:,:, 1:-1, 1:-1]

            def plot_a(i):
                plt.clf()
                plt.imshow(data[i, 0, :, :],
                           cmap='viridis')  # Adjust colormap as needed
                plt.colorbar(orientation='horizontal')
                plt.title(f'Layer 1, Frame {i}')

            def plot_p(i):
                plt.clf()
                plt.imshow(data[i, 1, :, :],
                           cmap='viridis')  # Adjust colormap as needed
                plt.colorbar(orientation='horizontal')
                plt.title(f'Layer 2, Frame {i}')

            # Animation for Layer 1
            print('plotting layer 1')
            fig1 = plt.figure()
            ani1 = FuncAnimation(fig1, plot_a, frames=num_frames,
                                 interval=50)  # Adjust interval as needed
            ani1.save(osp.join(store_dir, f'{fname}_vy_animation.gif'),
                      writer='imagemagick')

            # Animation for Layer 2
            print('plotting layer 2')
            fig2 = plt.figure()
            ani2 = FuncAnimation(fig2, plot_p, frames=num_frames,
                                 interval=50)  # Adjust interval as needed
            ani2.save(osp.join(store_dir, f'{fname}_vx_animation.gif'),
                      writer='imagemagick')

            
def plot_frames(data_dir, store_dir, num_frames=32, selected_frames=5):
    for f in tqdm(os.listdir(data_dir)):
        if f.endswith('.npy'):
            fname, fext = get_name(f)
            data = np.load(os.path.join(data_dir, f))
            data = data[:,:, 1:-1, 1:-1]
            # frame_indices = sorted(
            #    random.sample(range(data.shape[0]), selected_frames))
            frame_indices = frame_indices = [5, 15, 30]

            for layer in range(
                    data.shape[1]
            ):  # Assuming data shape is (frames, layers, height, width)
                for i in frame_indices:
                    plt.figure()
                    plt.imshow(data[i, layer, :, :],
                               cmap='viridis')  # Adjust colormap as needed
                    plt.colorbar(orientation='horizontal')
                    if layer == 0:
                        # switch to a / p if needed
                        plt.title(f'vy, Frame {i}')
                    else:  
                        plt.title(f'vx, Frame {i}')
                    plt.savefig(
                        osp.join(store_dir,
                                 f'{fname}_layer{layer+1}_frame{i}.jpg'))
                    plt.close()

if __name__ == "__main__":
    data_dir = '/project/t3_zxiaoal/Validation_Dataset/dt4_rlpinn_normalized_stable_v/airfoil149'
    store_dir = '/project/t3_zxiaoal/Validation_Dataset/visualizeresult/rlpinn_v'
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    plot_frames(data_dir, store_dir, num_frames=32)
    plot_gif(data_dir, store_dir, num_frames=32)
