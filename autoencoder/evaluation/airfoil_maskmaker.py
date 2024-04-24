# Make airfoil cond masks for airfoil type evaluation data

import torch
import numpy as np
import matplotlib.pyplot as plt

class MaskMaker:
    def __init__(self, airfoil_type, mask_shape, chord_length, origin, device = "cpu"):
        '''
        airfoil_type  : 4-digit airfoil type
        mask_shape    : (w, h)
        chord_length  : number of pixels along the chord
        origin        : (x, y), choordinate of the leading edge
        '''
        self.airfoil_type = airfoil_type
        self.mask_shape = mask_shape
        self.chord_length = chord_length
        self.origin = origin

        self.dx = 1 / self.chord_length
        self.dt = self.dx / 10 # Courant number of 0.1

        self.m, self.p, self.t = self.extract_parameters()

        if torch.cuda.is_available():
            device = device
        else:
            device = "cpu"

        self.device = torch.device(device)

    def extract_parameters(self):
        if len(self.airfoil_type) != 4:
            raise ValueError("Invalid airfoil type. Please provide a 4-digit NACA airfoil type.")

        m_str, p_str, t_str = self.airfoil_type[0], self.airfoil_type[1], self.airfoil_type[2:]

        try:
            m = int(m_str) / 100.0
            p = int(p_str) / 10.0
            t = int(t_str) / 100.0
        except ValueError:
            raise ValueError("Invalid airfoil type. Please provide a valid 4-digit NACA airfoil type.")

        return m, p, t

    def camber_line(self, x):
        if self.m == 0 or self.p == 0:
          return torch.zeros_like(x)
        return torch.where(x <= self.p, self.m / (self.p**2) * (2 * self.p * x - x**2), self.m / ((1 - self.p)**2) * ((1 - 2 * self.p) + 2 * self.p * x - x**2))

    def thickness_distribution(self, x):
        return 5 * self.t * (0.2969 * torch.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    def upper_surface(self, x):
        return self.camber_line(x) + self.thickness_distribution(x)

    def lower_surface(self, x):
        return self.camber_line(x) - self.thickness_distribution(x)

    def make_mask(self, rotation_angle=0):

        # Create a linear space
        dx = dy = 1 / self.chord_length
        x_lin = torch.linspace(-self.origin[0] * dx, (self.mask_shape[0] - self.origin[0]) * dx, self.mask_shape[0]).to(self.device)
        y_lin = torch.linspace(-self.origin[1] * dy, (self.mask_shape[1] - self.origin[1]) * dy, self.mask_shape[1]).flip(0).to(self.device)

        # Generate x, y values
        x, y = torch.meshgrid(x_lin, y_lin, indexing='xy')

        # Rotate the coordinates
        rotation_angle = -rotation_angle * torch.pi / 180
        x_t = x * torch.cos(torch.tensor(rotation_angle)) + y * torch.sin(torch.tensor(rotation_angle))
        y_t = -x * torch.sin(torch.tensor(rotation_angle)) + y * torch.cos(torch.tensor(rotation_angle))

        # Determine whether points are between upper and lower surfaces
        mask = torch.where((x_t >= 0) & (x_t <= 1) & (y_t >= self.lower_surface(x_t)) & (y_t <= self.upper_surface(x_t)), 1, 0).to(self.device, dtype=torch.float32)

        return mask

ref = torch.Tensor([[1.0000,0.0013],
  [0.9500,0.0147],
  [0.9000,0.0271],
  [0.8000,0.0489],
  [0.7000,0.0669],
  [0.6000,0.0814],
  [0.5000,0.0919],
  [0.4000,0.0980],
  [0.3000,0.0976],
  [0.2500,0.0941],
  [0.2000,0.0880],
  [0.1500,0.0789],
  [0.1000,0.0659],
  [0.0750,0.0576],
  [0.0500,0.0473],
  [0.0250,0.0339],
  [0.0125,0.0244],
  [0.0000,0.0000],
  [0.0125,-0.0143],
  [0.0250,-0.0195],
  [0.0500,-0.0249],
  [0.0750,-0.0274],
  [0.1000,-0.0286],
  [0.1500,-0.0288],
  [0.2000,-0.0274],
  [0.2500,-0.0250],
  [0.3000,-0.0226],
  [0.4000,-0.0180],
  [0.5000,-0.0140],
  [0.6000,-0.0100],
  [0.7000,-0.0065],
  [0.8000,-0.0039],
  [0.9000,-0.0022],
  [0.9500,-0.0016],
  [1.0000,-0.0013]])


mask_maker = MaskMaker("0012", (512, 512), 128, (192, 256))
np.save("path", mask_maker.make_mask(0).cpu())

