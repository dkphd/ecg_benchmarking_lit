import torch
import torch.nn as nn
import math

class GEV(nn.Module):
    def __init__(self):
        super(GEV, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))  # trainable parameter
        self.sigma = nn.Parameter(torch.tensor(1.0))  # trainable parameter
        self.xi = nn.Parameter(torch.tensor(0.0))  # trainable parameter

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=torch.finfo(self.sigma.dtype).eps)  # ensure sigma is positive

        # Type 1: For xi = 0 (Gumbel)
        def t1():
            return torch.exp(-torch.exp(-(x - self.mu) / sigma))

        # Type 2: For xi > 0 (Frechet) or xi < 0 (Reversed Weibull)
        def t23():
            y = (x - self.mu) / sigma
            y = self.xi * y
            y = torch.maximum(y, torch.tensor(-1.0))
            y = torch.exp(-torch.pow(torch.tensor(1.0) + y, -1.0 / self.xi))
            return y

        return torch.where(self.xi == 0, t1(), t23())


class mGEV(nn.Module):
    def __init__(self, num_classes):
        super(mGEV, self).__init__()
        self.mu = nn.Parameter(torch.zeros(num_classes))  # trainable parameter, initialized to 0
        self.sigma = nn.Parameter(torch.ones(num_classes))  # trainable parameter, initialized to 1
        self.xi = nn.Parameter(torch.tensor(0.1))  # trainable parameter, initialized to 0.1

    def forward(self, x):
        mu = self.mu
        sigma = torch.clamp(self.sigma, min=torch.finfo(self.sigma.dtype).eps)  # ensure sigma is positive
        xi = self.xi

        x = torch.clamp(x, -20, 20)  # clipping the inputs

        # Type 1: For xi = 0 (Gumbel)
        def t1():
            return torch.exp(-torch.exp(-(x - mu) / sigma))

        # Type 2: For xi > 0 (Frechet) or xi < 0 (Reversed Weibull)
        def t23():
            y = (x - mu) / sigma
            y = xi * y
            y = torch.maximum(y, torch.tensor(-1.0))
            y = torch.exp(-torch.pow(torch.tensor(1.0) + y, -1.0 / xi))
            return y

        mGEV = torch.where(xi == 0, t1(), t23())
        mGEV = mGEV / torch.sum(mGEV, dim=1, keepdim=True)  # Normalizing to make the sum 1
        return mGEV