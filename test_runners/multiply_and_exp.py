import torch

a = torch.rand(2,2)
print(a)
b = torch.pow(a, 2)
print(b)

print(torch.exp(b))
