import torch

cpu = torch.FloatTensor([1,2,3])
mps = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"

cpu2mps = cpu.to("mps")

print(cpu)
print(mps)
print(cpu2mps)

