import torch
from torch import nn
import numpy as np


def PINNLoss2(output, target, inputs, sevencutoff=1.7, zerocutoff=0.26, indexsmax=4, a=10 ** 4, b=10 ** 6, c=10 ** -6,
              d=10 ** -6):
    # Mean squared error
    loss = torch.nn.functional.mse_loss(output, target, reduction='mean')

    '''
    The PINN Loss must enforce the boundary conditions of the S-N curve.
    The S-N curve is a plot of stress amplitude vs. number of cycles to failure.
    The neural network outputs the number of cycles to failure given the stress max as input.

    Constraint 1: The first derivative of the S-N curve must be negative.
    Constraint 2: The second derivative of the S-N curve must be positive.
    Constraint 3: The S-N curve's slope must be 0 at 10^7 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 10^7 cycles.

    '''
    # Compute gradients
    # inputs.requires_grad_(True)
    outputmean = torch.mean(output)
    gradient1 = torch.autograd.grad(outputmean, inputs, create_graph=True)[0]

    # Constraint 1: The first derivative of the S-N curve must be negative.
    # Penalize positive first derivatives
    loss1 = a * torch.mean(torch.abs(torch.relu(gradient1[:, indexsmax])))
    # print(loss1)

    # Constraint 2: The second derivative of the S-N curve must be positive.
    # Compute second derivatives
    gradient2 = torch.autograd.grad(torch.mean(gradient1[:, indexsmax]), inputs, create_graph=True)[0]

    # Penalize negative second derivatives
    loss2 = b * torch.mean(torch.abs(torch.relu(-gradient2[:, indexsmax])))
    # print(loss2)

    # Constraint 3: The S-N curve's slope must be 0 at 10^7 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 10^7 cycles.
    # Penalize non-infinite gradients at 10^7 cycles
    loss3list = []
    for i in range(len(target)):
        if target[i] > sevencutoff:
            loss3list.append(gradient1[i, indexsmax])
    if len(loss3list) > 0:
        loss3 = c * torch.mean(torch.abs(1 / torch.tensor(loss3list)))
    else:
        loss3 = 0
    # print(loss3)

    loss4list = []
    for i in range(len(target)):
        if target[i] <= zerocutoff:
            loss4list.append(gradient1[i, indexsmax])
    if len(loss4list) > 0:
        loss4 = d * torch.mean(torch.abs(1 / torch.tensor(loss4list)))
    else:
        loss4 = 0
    # print(f"loss4 = {loss4}")

    return loss + loss3 + loss4 + loss1 + loss2


def MS(x):
    return torch.mean(x ** 2)


def Mexp(x, base=10):
    return torch.mean(base ** x)-1


def PINNLoss(output, target, inputs, sevencutoff=1.7, zerocutoff=0.3, indexsmax=4, f1=4 * 10 ** 5, f2=4 * 10 ** 6, f3=1 * 10 ** -5, f4=1 * 10 ** -5):
    error = torch.abs(target - output).view(-1)

    # Compute 1st and 2nd gradients
    gradient1 = torch.autograd.grad(torch.mean(output), inputs, create_graph=True)[0][:, indexsmax]
    gradient2 = torch.autograd.grad(torch.mean(gradient1), inputs, create_graph=True)[0][:, indexsmax]
    # Constraint 1: The first derivative of the S-N curve must be negative.
    # Penalize positive first derivatives
    loss1 = f1 * torch.relu(gradient1)

    # Constraint 2: The second derivative of the S-N curve must be positive.
    # Penalize negative second derivatives
    loss2 = f2 * torch.abs(torch.relu(-gradient2))

    # Constraint 3: The S-N curve's slope must be 0 at 0 and10^7 cycles. i.e. The output of the neural network (ncycles) must have a gradient of infinity with respect to the smax input at 10^7 cycles.
    # Penalize non-infinite gradients at high cycles
    loss3 = f3 * (target.view(-1) > sevencutoff) * torch.abs(1/gradient1)
    # penalize non-infinite gradients at low cycles
    loss4 = f4 * (target.view(-1) < zerocutoff) * torch.abs(1/gradient1)

    loss = error + loss1 + loss2 + loss3 + loss4
    return MS(loss)

class notMSELoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, reduction=self.reduction)


class altMSE(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        loss = torch.mean((1 + torch.relu(target) / 3) * (input - target) ** 2)
        return loss


class log_adjusted_MSE(nn.modules.loss._Loss):
    # mse = (log(p)-log(t))**2 => 10^sqrt(mse) = p/t
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        loss = torch.mean(torch.pow(10, torch.abs(input - target)))
        return loss
