import torch
from torch import nn
import numpy as np

def PINNLoss(output, target, inputs):
    # Mean squared error
    loss = torch.mean((output - target)**2)

    # Compute gradients
    inputs.requires_grad_(True)
    outputmean = torch.mean(output)
    gradient1 = torch.autograd.grad(outputmean, inputs, create_graph=True)[0]
    
    # Penalize positive first derivatives
    loss += torch.mean(torch.relu(gradient1[:, 8]))

    # Compute second derivatives
    gradient2 = torch.autograd.grad(torch.mean(gradient1[:, 8]), inputs, create_graph=True)[0]

    # Penalize negative second derivatives
    loss += torch.mean(torch.relu(-gradient2[:, 8]))

    return loss