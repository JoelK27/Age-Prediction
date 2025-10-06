import torch

def mae(p, t):
    return torch.mean(torch.abs(p - t))

def mse(p, t):
    return torch.mean((p - t) ** 2)