import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F


def get_name(path: str):
    '''input file path. return (fname, fext).'''
    return osp.splitext(osp.basename(osp.normpath(path)))



def rot_mac(a):
    return torch.cat([-dx_right(a), dy_bottom(a)], dim=1)


dx_right_kernel = torch.Tensor([0, -1, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(2)


def dx_right(v):
    return F.conv2d(v, dx_right_kernel, padding=(0, 1))


dy_bottom_kernel = torch.Tensor([0, -1, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(3)


def dy_bottom(v):
    return F.conv2d(v, dy_bottom_kernel, padding=(1, 0))


def get_v(data):
    data = torch.tensor(data)
    a = data[:, 0, :, :]
    v = []
    # iterate ofver data[:]
    for i in range(0, data.shape[0]):
        v.append(rot_mac(a[i].unsqueeze(0).unsqueeze(0)))
    return torch.stack(v).squeeze(1)

# Frequency difference between the two models

# data: (32, 2, 100, 300), 0: p; 1: a
def freq_and_L1_diff_ap(Overfitted, Trained, timestep=4):
    Overfitted_p = Overfitted[:, 0].mean(axis=(-1, -2))
    Trained_p = Trained[:, 0].mean(axis=(-1, -2))
    Overfitted_a = Overfitted[:, 0].mean(axis=(-1, -2))
    Trained_a = Trained[:, 0].mean(axis=(-1, -2))
    
    Overfitted_p_mean = Overfitted_p.mean()
    Trained_p_mean = Trained_p.mean()
    Overfitted_a_mean = Overfitted_a.mean()
    Trained_a_mean = Trained_a.mean()
    
    Overfitted_p = Overfitted_p - Overfitted_p_mean
    Trained_p = Trained_p - Trained_p_mean
    Overfitted_a = Overfitted_a - Overfitted_a_mean
    Trained_a = Trained_a - Trained_a_mean
    
    Overfitted_v = get_v(Overfitted)
    Trained_v = get_v(Trained)
    Overfitted_U = torch.sqrt(Overfitted_v[:, 0]**2 + Overfitted_v[:, 1]**2).numpy()
    Trained_U = torch.sqrt(Trained_v[:, 0]**2 + Trained_v[:, 1]**2).numpy()
    Overfitted_U = Overfitted_U.mean(axis=(-1, -2))
    Trained_U = Trained_U.mean(axis=(-1, -2))
    
    Overfitted_U_mean = Overfitted_U.mean()
    Trained_U_mean = Trained_U.mean()
    
    Overfitted_U = Overfitted_U - Overfitted_U_mean
    Trained_U = Trained_U - Trained_U_mean

    # fft on p and U
    Overfitted_p_fft = np.fft.fft(Overfitted_p)
    Trained_p_fft = np.fft.fft(Trained_p)
    Overfitted_U_fft = np.fft.fft(Overfitted_U)
    Trained_U_fft = np.fft.fft(Trained_U)
    
    frequencies = np.fft.fftfreq(Overfitted_p.size, timestep)
    positive_frequencies = frequencies[:frequencies.size//2]
    
    amplitude_spec_Overfitted_p = np.sqrt(np.abs(Overfitted_p_fft[:Overfitted_p_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Trained_p = np.sqrt(np.abs(Trained_p_fft[:Trained_p_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Overfitted_U = np.sqrt(np.abs(Overfitted_U_fft[:Overfitted_U_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Trained_U = np.sqrt(np.abs(Trained_U_fft[:Trained_U_fft.size//2]))* 2 / Overfitted_p.size
    
    max_freq_Overfitted_p = positive_frequencies[np.argmax(amplitude_spec_Overfitted_p)]
    max_freq_Trained_p = positive_frequencies[np.argmax(amplitude_spec_Trained_p)]
    max_freq_Overfitted_U = positive_frequencies[np.argmax(amplitude_spec_Overfitted_U)]
    max_freq_Trained_U = positive_frequencies[np.argmax(amplitude_spec_Trained_U)]
    
    p_freq_diff = np.abs(max_freq_Overfitted_p - max_freq_Trained_p)
    U_freq_diff = np.abs(max_freq_Overfitted_U - max_freq_Trained_U)
    
    p_mean_diff = np.abs(Overfitted_p_mean - Trained_p_mean)
    U_mean_diff = np.abs(Overfitted_U_mean - Trained_U_mean)
    
    return p_freq_diff, U_freq_diff, p_mean_diff, U_mean_diff


def freq_and_L1_diff(Overfitted, Trained, timestep=4):
    Overfitted_p = Overfitted[:, 0].mean(axis=(-1, -2))
    Trained_p = Trained[:, 0].mean(axis=(-1, -2))
    Overfitted_a = Overfitted[:, 1].mean(axis=(-1, -2))
    Trained_a = Trained[:, 1].mean(axis=(-1, -2))
    
    Overfitted_p_mean = Overfitted_p.mean()
    Trained_p_mean = Trained_p.mean()
    Overfitted_a_mean = Overfitted_a.mean()
    Trained_a_mean = Trained_a.mean()
    
    Overfitted_p = Overfitted_p - Overfitted_p_mean
    Trained_p = Trained_p - Trained_p_mean
    Overfitted_a = Overfitted_a - Overfitted_a_mean
    Trained_a = Trained_a - Trained_a_mean
    
    # fft on p and U
    Overfitted_p_fft = np.fft.fft(Overfitted_p)
    Trained_p_fft = np.fft.fft(Trained_p)
    Overfitted_a_fft = np.fft.fft(Overfitted_a)
    Trained_a_fft = np.fft.fft(Trained_a)
    
    frequencies = np.fft.fftfreq(Overfitted_p.size, timestep)
    positive_frequencies = frequencies[:frequencies.size//2]
    
    amplitude_spec_Overfitted_p = np.sqrt(np.abs(Overfitted_p_fft[:Overfitted_p_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Trained_p = np.sqrt(np.abs(Trained_p_fft[:Trained_p_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Overfitted_a = np.sqrt(np.abs(Overfitted_a_fft[:Overfitted_a_fft.size//2]))* 2 / Overfitted_p.size
    amplitude_spec_Trained_a = np.sqrt(np.abs(Trained_a_fft[:Trained_a_fft.size//2]))* 2 / Overfitted_p.size
    
    max_freq_Overfitted_p = positive_frequencies[np.argmax(amplitude_spec_Overfitted_p)]
    max_freq_Trained_p = positive_frequencies[np.argmax(amplitude_spec_Trained_p)]
    max_freq_Overfitted_a = positive_frequencies[np.argmax(amplitude_spec_Overfitted_a)]
    max_freq_Trained_a = positive_frequencies[np.argmax(amplitude_spec_Trained_a)]
    
    p_freq_diff = np.abs(max_freq_Overfitted_p - max_freq_Trained_p)
    a_freq_diff = np.abs(max_freq_Overfitted_a - max_freq_Trained_a)
    
    p_mean_diff = np.abs(Overfitted_p_mean - Trained_p_mean)
    a_mean_diff = np.abs(Overfitted_a_mean - Trained_a_mean)
    
    return p_freq_diff, a_freq_diff, p_mean_diff, a_mean_diff