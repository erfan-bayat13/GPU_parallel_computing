import torch
import transpose_ext

A = torch.arange(6, device="cuda").reshape(2, 3).float()
B = transpose_ext.transpose_forward(A)

print("A:\n", A)
print("B:\n", B)
print("Matches PyTorch:", torch.allclose(B, A.T))
