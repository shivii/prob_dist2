import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class SGAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGAT, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(out_channels*2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj_matrix):
        # x: input feature matrix [batch_size, num_nodes, in_channels]
        # adj_matrix: adjacency matrix [batch_size, num_nodes, num_nodes]
        
        # compute node features using convolution
        h = self.conv(x)
        
        # compute attention weights between nodes
        a = torch.matmul(h, h.permute(0, 2, 1))
        a = self.attention(torch.cat([h.unsqueeze(2).repeat(1,1,a.size(2),1), 
                                       h.unsqueeze(1).repeat(1,a.size(1),1,1)], dim=-1))
        a = a * adj_matrix
        
        # normalize attention weights and apply to neighbor node features
        a = a / (torch.sum(a, dim=-1, keepdim=True) + 1e-10)
        h = torch.matmul(a, h)
        
        return h, a
        
# create SGAT model
sgat = SGAT(in_channels=3, out_channels=64)

# load image and convert to feature matrix
img_path = "/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_real_B.png"
img = Image.open(img_path)
img = img.resize((224,224))
x = torch.Tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)

# create adjacency matrix
num_nodes = x.size(2) * x.size(3)
adj_matrix = torch.zeros(1, num_nodes, num_nodes)
for i in range(num_nodes):
    for j in range(i, num_nodes):
        if i == j:
            adj_matrix[0, i, j] = 1
        else:
            x1, y1 = i // x.size(3), i % x.size(3)
            x2, y2 = j // x.size(3), j % x.size(3)
            if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                adj_matrix[0, i, j] = 1
                adj_matrix[0, j, i] = 1

print("adj matrix :", adj_matrix.shape)

# run image through SGAT to get adjacency matrix and node features
h, a = sgat(x, adj_matrix)

# use adjacency matrix as input to GAE model
