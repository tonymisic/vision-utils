import torch, seaborn as sns, maths
from sklearn.manifold import TSNE
class Visualization:
    ''' Class containing tensor visualization methods
    '''
    def flatten_data(data, spatial_labels, temporal_labels):
        augmented = torch.zeros([data.size(0) * data.size(1), data.size(2)])
        labels = torch.zeros([data.size(0) * data.size(1)])
        for i in range(data.size(0)): # dataset length
            for j in range(data.size(1)): # 10
                augmented[i * data.size(1) + j] = data[i, j]
                if temporal_labels[i, j] == 1:
                    labels[i * data.size(1) + j] = spatial_labels[i]
        return augmented, labels

    def tsne(self, data, labels, savefile="tsne.jpg", n_comps=2, n_iters=250):
        assert data.size(0) == labels.size(0)
        t = TSNE(n_components=n_comps, n_iter=n_iters)
        x = t.fit_transform(X=data)
        plot = sns.scatterplot(x[:,0], x[:,1], hue=labels, legend='full')
        figure = plot.get_figure()
        figure.savefig(savefile)
        print("Saved t-SNE visualization to: " + savefile)
    
    def per_class_variance(data, labels, num_classes):
        assert data.size(0) == labels.size(0)
        variances = []
        # organize data into classes
        for sample in range(num_classes):
            variances.append(float(maths.variance(maths.variance(data[labels == sample]))))
        return variances
    
    def visualize_variances(variances, savefile="var.jpg"):
        plot = sns.barplot(x=torch.range(0, len(variances) - 1, dtype=int).tolist(), y=variances)
        figure = plot.get_figure()
        figure.savefig(savefile)
        print("Saved Class Variance visualization to: " + savefile)

