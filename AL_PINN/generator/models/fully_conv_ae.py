import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Define the Convolutional Autoencoder model
class FullyConvAE(nn.Module):
    def __init__(self):
        super(FullyConvAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 3, 3), stride=(1, 1, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(128, 256, kernel_size=(3, 7, 7)),

        )

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8,1), stride=(4,1), padding=(2, 0)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8,1))
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=7),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=(1,3), padding=(1,0)),
            nn.Sigmoid()
        )
        
        self.mean = torch.load('pretrain_weights/mean.pt')
        self.std = torch.load('pretrain_weights/std.pt')
        self.mean = self.mean.reshape(1, 256, 1, 1)
        self.std = self.std.reshape(1, 256, 1, 1)
    
    def forward(self, x):
        split_input = torch.chunk(x, 32)

        encoded = []
        for tensor in split_input:
            encoded_tensor = self.encoder(tensor)
            encoded.append(encoded_tensor)
        encoded = torch.stack(encoded).unsqueeze(0).squeeze(3).squeeze(3).squeeze(3)

        adapted = self.adapter(encoded)
        adapted = adapted.unsqueeze(0)
        adapted =adapted.permute(0, 3, 1, 2)
        adapted_con_mask = adapted #256
        adapted_v = torch.rand(1,1,1,1) * 2 - 1 # 256
        mask_mean = torch.mean(adapted_con_mask)
        mask_std = torch.std(adapted_con_mask)
        adapted_con_mask = (adapted_con_mask - mask_mean) / mask_std
        adapted_con_mask = adapted_con_mask * self.std + self.mean
        output = self.decoder(adapted_con_mask)
        entrance_mask = torch.zeros(1, 1, 100, 300).to('cuda')

        entrance_mask[:, :, 3:-3, 5:-5] = 1

        added_entrance = 1 - (1 - output) * entrance_mask
        vx = torch.zeros(1, 1, 100, 300).to('cuda')
        vy = torch.zeros(1, 1, 100, 300).to('cuda')
        vx[0,0,3:-3,0:5] += adapted_v[0,0,0,0]
        vx[0,0,3:-3,-5:] += adapted_v[0,0,0,0]
        output = torch.cat((added_entrance, vx, vy), dim=1)
        return output