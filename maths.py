import torch, math, torch.nn as nn
def variance(tensors):
    return torch.var(tensors, 0, unbiased=False)

def mean(tensors):
    return torch.mean(tensors, dim=0)

# https://en.wikipedia.org/wiki/Bhattacharyya_distance
def bhattacharyya(sample1, sample2):
    sigma1, sigma2 = variance(variance(sample1)), variance(variance(sample2))
    mu1, mu2 = mean(mean(sample1)), mean(mean(sample2))
    return 0.25 * math.log(0.25 * (sigma2 / sigma1 + sigma1 / sigma2 + 2)) + (0.25 * (math.pow(mu2 - mu1, 2) / (sigma1 + sigma2)))