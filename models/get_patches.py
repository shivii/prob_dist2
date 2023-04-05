# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

#reshape patch
def resize_patch(x, patch_size):
  #print("size of Patch :", x.shape)
  size = int(3*patch_size*patch_size)
  result = x.reshape(1,size)
  #print(result.shape)
  return result


def get_patch_list(image, label, patch_size):
    #print("----------------------getting neighbours list-------------------------")
    image = torch.squeeze(image)
    #print("image:", image.shape)
    #transt = transforms.ToTensor()
    #transp = transforms.ToPILImage()
    #image = transt(image)
    shape_i, shape_j, shape_k  = image.shape
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(3*patch_size*patch_size)
    
    label = label
    features = torch.zeros((0,size), dtype=torch.float32)
    labels = torch.zeros((0,1), dtype=torch.float32)
    
    size = patch_size
    #torch.Tensor.unfold(dimension, size, step)
    #slices the images into 8*8 size patches
    patches = image.data.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
    #print(patches.shape)
    count = 0
    total_patches_x = int(256/(patch_size))
    for i in range(total_patches_x):
        for j in range(total_patches_x):
            #print(patches[0][i][j].size)
            resized_patch = resize_patch(patches[0][i][j], patch_size)
            #print(resized_patch.shape)
            features = torch.cat((features, resized_patch),0)
            labels = torch.cat((labels, label), 0)
            count = count + 1
    #print("Total patches:", count)

    #print("features, labels shape :", features.shape, labels.shape)
            
    #print("returning list")
    return features, labels