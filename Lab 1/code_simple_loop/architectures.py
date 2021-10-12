# Specification of Parametrization class

import torch
import torch.nn as nn

class Parametrization(nn.Module):
    
    # The function that specifies dara initalization.
    # Called when creating object of Parametrization class
    def __init__(self, inputDim, outputDim):
        
        # Initialize parent:
        super().__init__()
        
        # Definition of the tensor where weights are stored
        self.H = nn.parameter.Parameter(torch.zeros(inputDim, outputDim))            
    
    # Forward method. This is how the parametrization predicts from inputs
    def forward(self, x):
        
        # Calculate output estimate
        yHat = torch.matmul(x,self.H)

        return yHat
