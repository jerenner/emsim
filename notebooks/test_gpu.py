import torch
cuda = torch.device('cuda')
n = 1000000000

i = 0
while i < n:
    tensor1 = torch.randn(1000, 1000, device=cuda)
    tensor2 = torch.randn(1000, 1000, device=cuda)
    torch.cuda.synchronize()
    x = torch.matmul(tensor1, tensor2)
    i += 1

print("finished")
