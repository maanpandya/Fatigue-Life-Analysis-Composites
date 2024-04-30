import torch
from torch import nn
import numpy as np

def PINNLoss(output, target, inputs):
    # Mean squared error
    loss = torch.mean((output - target)**2)

    '''
    The PINN Loss must enforce the boundary conditions of the S-N curve.
    The S-N curve is a plot of stress amplitude vs. number of cycles to failure.
    The neural network outputs the number of cycles to failure given the stress max as input.

    Constraint 1: The first derivative of the S-N curve must be negative.
    Constraint 2: The second derivative of the S-N curve must be positive.
    Constraint 3: The S-N curve's slope must be 0 at 0 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 0 cycles.
    Constraint 4: The S-N curve's slope must be 0 at 10^7 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 10^7 cycles.

    '''
    indexsmax = 8
    a=10000
    b=10000
    c=10000

    # Compute gradients
    inputs.requires_grad_(True)
    outputmean = torch.mean(output)
    gradient1 = torch.autograd.grad(outputmean, inputs, create_graph=True)[0]
    
    #Constraint 1: The first derivative of the S-N curve must be negative.
    # Penalize positive first derivatives
    loss1 = a*torch.mean(abs(torch.relu(gradient1[:, indexsmax])))
    #print(loss1)
    loss += loss1

    #Constraint 2: The second derivative of the S-N curve must be positive.
    # Compute second derivatives
    gradient2 = torch.autograd.grad(torch.mean(gradient1[:, indexsmax]), inputs, create_graph=True)[0]

    # Penalize negative second derivatives
    loss2 = b*torch.mean(abs(torch.relu(-gradient2[:, indexsmax])))
    #print(loss2)
    loss += loss2

    #Constraint 3: The S-N curve's slope must be 0 at 0 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 0 cycles.
    # Penalize non-infinite gradients at 0 cycles
    loss3 = c*torch.mean(abs(1/(np.array([i for i in gradient1[:, indexsmax] if i == 0]))))
    #print(loss3)
    loss += loss3

    #Constraint 4: The S-N curve's slope must be 0 at 10^7 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 10^7 cycles.
    # Penalize non-infinite gradients at 10^7 cycles
    loss4 = c*torch.mean(abs(1/(np.array([i for i in gradient1[:, indexsmax] if i >= 10000000]))))
    #print(loss4)
    loss += loss4
    
    return loss