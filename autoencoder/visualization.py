import numpy as np
import torch
from PIL import Image
import imageio

tensor = torch.load("/data/rliuak/resnet_autodecoder/autoencoder/param_search_log/search_1/a_log/a_new7.pth").cpu()
tensor = tensor.detach().numpy()
print(tensor.shape)
tensor = np.reshape(tensor, (64, 100, 300))
tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

frames = []
for t in range(64):
    frame = tensor[t]
    frame = np.uint8(frame * 255)
    frame = Image.fromarray(frame, mode='L')
    frames.append(frame)

imageio.mimsave('animation.gif', frames, format='GIF', duration=0.1)