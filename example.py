from file_io import data_from_h5file
from visualization import Visualization
import torch, maths
ROOT = 'AVE_Dataset/'
# video_data = torch.from_numpy(data_from_h5file(ROOT + 'visual_feature.h5'))
audio_data = torch.from_numpy(data_from_h5file(ROOT + 'audio_feature.h5'))
spatial_labels = torch.from_numpy(data_from_h5file(ROOT + 'labels.h5'))
temporal_labels = torch.from_numpy(data_from_h5file(ROOT + 'labels_closs.h5'))
data, labels = Visualization.flatten_data(audio_data, spatial_labels, temporal_labels)

distances = Visualization.bhattacharyya_distances(data, labels, 29)
Visualization.visualize_bhattacharyya(distances)
exit()
variances = Visualization.per_class_variance(data, labels, 29)
Visualization.visualize_variances(variances, savefile="audio_var.jpg")
video_data = torch.flatten(video_data, start_dim=2, end_dim=4)
data, labels = Visualization.flatten_data(video_data, spatial_labels, temporal_labels)
variances = Visualization.per_class_variance(data, labels, 29)
Visualization.visualize_variances(variances, savefile="video_var.jpg")