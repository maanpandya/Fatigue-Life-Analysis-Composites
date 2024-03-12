import torch
import numpy as np
def pinnloss(output,target):
    loss = torch.mean((output-target)**2)
    return loss