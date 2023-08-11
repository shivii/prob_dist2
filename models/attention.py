import torch
import torch.nn as nn

# Define the self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.W_k = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        attention_scores = torch.nn.functional.softmax(torch.matmul(query.view(query.size(0), -1, query.size(2)).transpose(1, 2),
                                                                     key.view(key.size(0), -1, key.size(2))), dim=-1)
        attended_values = torch.matmul(attention_scores, value.view(value.size(0), -1, value.size(2)))
        attended_values = attended_values.view(value.size())
        
        return x + attended_values