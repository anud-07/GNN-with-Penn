# Specification of Parametrization class

import torch
import torch.nn as nn

class TwoLayerNN(nn.Module):

    # The function that specifies data initalization.
    # Called when creating object of TwoLayersNN class
    def __init__(self, n, m, h):

        # Initialize parent:
        super().__init__()

        # Definition of the tensor where weights are stored
        self.H1 = nn.parameter.Parameter(torch.rand(n, h))
        self.H2 = nn.parameter.Parameter(torch.rand(h, m))

    # Forward method. This is how the parametrization predicts from inputs
    def forward(self, x):
        sigma = nn.ReLU()
        z = sigma(torch.matmul(x,self.H1))
        yHat = torch.matmul(z,self.H2)
        return yHat


