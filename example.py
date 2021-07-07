'''
Examples of using code
'''
from numpy.random import sample
import torch
from visualization import Visualization

sample_video = torch.rand(3, 10, 25088)
sample_audio = torch.rand(3, 10, 128)
temporal_labels = torch.zeros([3, 10])
temporal_labels[0, 2:5] = 1
temporal_labels[1, 3:9] = 1
temporal_labels[2, 2:7] = 1
spatial_labels = torch.Tensor([3, 1, 2])
data, labels = Visualization.flatten_data(sample_audio, spatial_labels, temporal_labels)
Visualization.per_class_variance(data, labels, 4)