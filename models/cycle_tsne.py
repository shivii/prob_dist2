# -*- coding: utf-8 -*-
#from sklearn.manifold import TSNE
import torch
from tsnecuda import TSNE 
from sklearn.manifold import TSNE as TSNE_sk
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from statistics import mean

if __name__ == '__main__':
    from utility import print_with_time as print
else:
    from models.utility import print_with_time as print

def get_tsne(data, labels):  
    n_components = 2
    #tsne = TSNE(n_components = n_components, random_state = 0)
    #tsne_data = tsne.fit_transform(data)
    print("computing TSNE for data with size:", data.shape)
    tsne_data = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(data)
    print("Computed TSNE data has size:", tsne_data.shape)
    return tsne_data

def get_tsne_sk(data, labels):
    n_components = 2
    print("computing TSNE for data with size:", data.shape)
    tsne = TSNE_sk(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    print("Computed TSNE data has size:", tsne_data.shape)
    return tsne_data

def plot_representations(tx, ty, labels, name):
    classes = [1,0]
    # initialize a matplotlib plot
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # for every class, we'll add a scatter plot separately
    for label in classes:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        #color = np.array(classes[label], dtype=np.float) / 255
        #color = range(len(targets))
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plot_name = "projection_" + name
    plt.savefig(plot_name)    
    # finally, show the plot
    #plt.show()

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def tsne_loss(tx, ty):
    print("Computing TSNE loss")
    distances = []
    for i in range(0, len(tx), 2):
        p1 = (tx[i], ty[i]) # first point
        p2 = (tx[i+1], ty[i+1]) # second point
        dist = sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # Pythagorean theorem
        distances.append(dist)
    distance = mean(distances)
    print("computed TSNE distances", distance.shape)
    return distance

if __name__ == '__main__':
    data0 = torch.rand(4096, 48)
    data = torch.rand(4096, 48)
    data1 = torch.rand(4096, 3072)
    get_tsne(data0, 1)
    get_tsne(data, 1)
    get_tsne(data1, 1)
    #get_tsne_sk(data, 1)
