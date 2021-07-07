import torch, seaborn as sns
from sklearn.manifold import TSNE
class Visualization:
    ''' Class containing tensor visualization methods
    '''
    def __init__(self):
        return 1
    
    # t-SNE of a tensor (samples, size) and labels (samples)
    def tsne(tensor, labels, savefile="tsne.jpg", n_comps=2, n_iters=100):
        assert tensor.size(0) == labels.size(0)
        t = TSNE(n_components=n_comps, n_iter=n_iters)
        embedded = t.fit_transform(tensor)
        plot = sns.scatterplot(embedded[:,0], embedded[:,1], hue=labels, legend='full')
        plot.savefig(savefile)
        print("Saved t-SNE visualization to: " + savefile)
    
    # Visualize per class grouping
    
    # Visualize per modality class grouping