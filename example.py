'''
Examples of using code
'''
import torch
from visualization import Visualization

sample_video = torch.rand(10, 512, 7, 7)
sample_audio = torch.rand(10, 128)
temporal_label = torch.zeros([10])
temporal_label[3:10] = 1
spatial_label = torch.zeros([10])
Visualization.tsne(sample_audio, temporal_label)