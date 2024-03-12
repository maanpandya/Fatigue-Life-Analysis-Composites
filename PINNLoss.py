import torch
import numpy as np
def pinnloss(output,target,inputs):
    #Mean squared error
    loss = torch.mean((output-target)**2)
    #Penalty if the first derivative of the output with respect to 'smax' (7th column, 6th index) is positive
    stress = inputs[:,6]
    stress.requires_grad = True
    gradient = torch.autograd.grad(output,stress,retain_graph=True)[0]
    #Penalize positive gradients
    loss += torch.mean(torch.relu(gradient))
    return loss