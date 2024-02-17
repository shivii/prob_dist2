# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
import math

if __name__ == '__main__':
    from utility import print_with_time as print
else:
    from models.utility import print_with_time as print

def get_neighb_list(image, label):
    print("----------------------getting neighbours list-------------------------")
    image = torch.squeeze(image)
    print("image:", image.shape)
    shape_i, shape_j, shape_k  = image.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    label = label.to(device)
    features = torch.zeros((0,6), dtype=torch.float32).to(device)
    labels = torch.zeros((0,1), dtype=torch.float32).to(device)
    for i, x in enumerate(image):
        #print("x1", x.shape)
        #print("----------------------computing neighbours")
        for j, x in enumerate(x):
            #print("x2", x.shape)
            for k, x in enumerate(x):
                #print("-->>---x3-------------------------------------------------------", x)
                feature = torch.zeros((0), dtype=torch.float32).to(device)
                fr = torch.zeros((1), dtype=torch.float32)
                up = torch.zeros((1), dtype=torch.float32)
                lf = torch.zeros((1), dtype=torch.float32)
                rt = torch.zeros((1), dtype=torch.float32)
                dw = torch.zeros((1), dtype=torch.float32)
                bc = torch.zeros((1), dtype=torch.float32)
                
                if i == 0:
                  fr = fr.to(device)
                  #print("shape fr:", fr, fr.shape)
                else:
                  fr = torch.unsqueeze(image[i-1,j,k], 0)
                  #print("shape fr:", fr, fr.shape)
                  
                feature = torch.cat((feature, fr), 0)
                
                if j == 0:
                  up = up.to(device)
                  #print("shape up:", up , up.shape)
                else:
                  up = torch.unsqueeze(image[i,j-1,k], 0)
                  #print("shape up:", up, up.shape)
                  
                feature = torch.cat((feature, up), 0)
                
                if k == 0:
                  lf = lf.to(device)
                  #print("shape lf:", lf, lf.shape)
                else:
                  lf = torch.unsqueeze(image[i,j,k-1], 0)
                  #print("shape lf:", lf, lf.shape)
                  
                feature = torch.cat((feature, lf), 0)
                
                if k+1 > shape_j-1:
                  rt = rt.to(device)
                  #print("shape rt:", rt, rt.shape)
                else:
                  rt = torch.unsqueeze(image[i,j,k+1], 0)
                  #print("shape rt:", rt, rt.shape)
                
                feature = torch.cat((feature, rt), 0)
                
                if j+1 > shape_j-1:
                  dw = dw.to(device)
                  #print("shape dw:", dw, dw.shape)
                else:
                  dw = torch.unsqueeze(image[i,j+1,k], 0)
                  #print("shape dw:", dw, dw.shape)
                  device = "cuda" if torch.cuda.is_available() else "cpu"

                feature = torch.cat((feature, dw), 0)
                
                if i+1 > shape_i-1:
                  bc = bc.to(device)
                  #print("shape bc:", bc, bc.shape)
                else:
                  bc = torch.unsqueeze(image[i+1, j, k], 0)
                  #print("shape bc:", bc, bc.shape)
                  
                feature = torch.cat((feature, bc), 0)
                feature = torch.unsqueeze(feature, 0)
                #print("feature:", feature, feature.shape)
                #feature = torch.cat((fr, up, lf, rt, dw, bc))                
                features = torch.cat((features, feature),0)
                labels = torch.cat((labels, label), 0)
    print("features, labels shape :", features.shape, labels.shape)
    
    print("returning list")
    return features, labels

import numpy as np


def get_neighb_gaussian_numpy_impl(image_tensor, sigma):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.squeeze(image_tensor).detach().cpu().numpy()
      
    print(image.shape)
    
    depth, rows, columns = image.shape
      
    print("started whatever")
    neighbour_list = []
    for r in range(rows):
        for c in range(columns):
            for d in range(depth):
                neighbours = get_n_hop_neighbours_gaussian(image, r,c,d, 2, sigma)
                neighbour_list.append(neighbours)
      
    print("completed numpy impl converting to tensor now")
    neighbour_tensor = torch.from_numpy(np.array(neighbour_list)).to(device)
    print("converted to tensor now")
    return neighbour_tensor


"""
image is the image in which the neihbour is being computed
r, c, d are row column and depth of the pixel for which
the neihbours are being computed
n is the number of hops at which neihbours need to be found n >=1
"""
def get_n_hop_neighbours_gaussian(image, r, c, d, n):
    rows, columns, depth = image.shape
    value_at_pixel = image[d][r][c]
    
    start_row = r-n
    end_row = r+n
    start_column = c-n
    end_column = c+n
    
    neighbours = []
    
    for row_iterator in range(start_row, end_row+1):
        for col_iterator in range(start_column, end_column+1):
            if row_iterator == r and col_iterator == c:
                ## skip the pixel for which neighbours are being computed
                continue
            neighbour_distance = 0
            if (row_iterator >= 0 and row_iterator < rows and col_iterator >= 0 and col_iterator < columns):
                difference = abs(value_at_pixel - image[d][row_iterator][col_iterator])
                exponent = -math.pow(difference, 2)/math.pow(sigma, 2)
                neighbour_distance = math.exp(exponent)
            neighbours.append(neighbour_distance)
    return neighbours

def get_neighb_numpy_impl(image_tensor):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  image = torch.squeeze(image_tensor).detach().cpu().numpy()

  print(image.shape)
  
  depth, rows, columns = image.shape

  print("started whatever")
  neighbour_list = []
  for r in range(rows):
    for c in range(columns):
      for d in range(depth):
        neighbours = get_n_hop_neighbours(image, r,c,d, 2)
        neighbour_list.append(neighbours)

  print("completed numpy impl converting to tensor now")
  neighbour_tensor = torch.from_numpy(np.array(neighbour_list)).to(device)
  print("converted to tensor now")
  return neighbour_tensor

"""
image is the image in which the neihbour is being computed
r, c, d are row column and depth of the pixel for which
the neihbours are being computed
n is the number of hops at which neihbours need to be found n >=1
"""
def get_n_hop_neighbours(image, r, c, d, n):
  rows, columns, depth = image.shape
  value_at_pixel = image[d][r][c]
  
  start_row = r-n
  end_row = r+n
  start_column = c-n
  end_column = c+n

  neighbours = []

  for row_iterator in range(start_row, end_row+1):
    for col_iterator in range(start_column, end_column+1):
      if row_iterator == r and col_iterator == c:
        ## skip the pixel for which neighbours are being computed
        continue
      difference = 0
      if (row_iterator >= 0 and row_iterator < rows and col_iterator >= 0 and col_iterator < columns):
              difference = abs(value_at_pixel - image[d][row_iterator][col_iterator])
      
      neighbours.append(difference)

  return neighbours


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("started", device)
    real = torch.tensor([[1],]).to(device)
    fake = torch.tensor([[0],]).to(device)
    data0 = torch.rand(1, 256,256,3).to(device)
    print("calling get neighbours")
    val = get_neighb_numpy_impl(data0)
    print(val.shape)
    print("done with get neighbours")
    

    
