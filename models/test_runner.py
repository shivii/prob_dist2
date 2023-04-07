# -*- coding: utf-8 -*-

import torch
import get_neigh
import cycle_tsne
from utility import print_with_time as print

if __name__ == '__main__':
    print("started the runner will check device")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device is ", device)
    print("generating random tensors")
    data0 = torch.rand(1, 256,256,3).to(device)
    data1 = torch.rand(1, 256,256,3).to(device)
    print("calling get neighbours for data0")
    val0 = get_neigh.get_neighb_numpy_impl(data0)
    print(val0.shape)
    print("done with get neighbours for data0")
    print("calling get neighbours for data1")
    val1 = get_neigh.get_neighb_numpy_impl(data1)
    print(val1.shape)
    print("done with get neighbours for data1")
    
    
    print("calling tsne with val0")
    tsne_data0 = cycle_tsne.get_tsne(val0, 1)
    print(tsne_data0.shape)
    print("done with tsne for val0")
    
    print("calling tsne with val1")
    tsne_data1 = cycle_tsne.get_tsne(val1, 1)
    print(tsne_data1.shape)
    print("done with tsne for val1")
