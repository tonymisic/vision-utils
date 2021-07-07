import torch
def variance(tensors):
        return torch.var(tensors, 0, unbiased=False)