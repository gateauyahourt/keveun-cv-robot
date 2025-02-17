import torch

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps")
else:
    print("MPS is not available")
    device = torch.device("cpu")