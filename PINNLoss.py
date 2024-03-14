import torch
import numpy as np
'''def PINNLoss(output,target,inputs):
    #Mean squared error
    loss = torch.mean((output-target)**2)

    #Penalty if the first derivative of the output with respect to 'smax' (9th column, 8th index) is positive
    inputs.requires_grad = True
    outputmean = torch.mean(output)
    outputmean.backward(retain_graph=True)
    #print(inputs.grad)
    gradient1 = inputs.grad[:,8]

    #Penalize positive gradients
    loss += torch.mean(torch.relu(gradient1))

    #Penalty if the second derivative of the output with respect to 'smax' is negative
    gradient1.backward(retain_graph=True)
    gradient2 = inputs.grad[:,8]
    #print(inputs.grad)

    #Penalize negative gradients
    loss += torch.mean(torch.relu(-gradient2))

    return loss'''

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