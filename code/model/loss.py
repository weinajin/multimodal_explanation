import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target)

def ce_loss_weighted(output, target):
    weighted_loss = torch.tensor([0.34, 0.66])
    return F.cross_entropy(output, target, weight= weighted_loss)

def focal_loss(output, target, gama=2., size_average=True, weight =None):
    log_P = -F.cross_entropy(output, target, weight=self.weight, reduction='none')
    P = torch.exp(log_P)
    batch_loss = -torch.pow(1 - P, gama).mul(log_P)
    if size_average:
        loss = batch_loss.mean()
    else:
        loss = batch_loss.sum()
    return loss

