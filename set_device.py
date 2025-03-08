import torch

device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
cpu = torch.FloatTensor([1,2,3])
#gpu = torch.cuda.FloatTensor([1,2,3])
tensor = torch.rand((1,1), device=device)

#print(torch.cuda.is_available())

print(device)
print(cpu)
#print(gpu)
print(tensor)
