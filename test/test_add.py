import flaggems._C
import torch

x = torch.rand(1111111111, device="cuda")
y = torch.rand(1111111111, device="cuda")
z = torch.rand(1111111111, device="cuda")
print(z)
z = flaggems._C.add(x, y, alpha=1, out=z)
print(x)
print(y)
print(z)
