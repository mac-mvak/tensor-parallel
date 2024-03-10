import torch
import torch.distributed as dist
from torch.autograd import Function


class gather1(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, rank: int, size: int):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`

        rank = torch.tensor(rank, dtype=int, device=input.device)
        size = torch.tensor(size, dtype=int, device=input.device)

        ctx.save_for_backward(rank, size)

        tensor_list = [torch.empty_like(input) for _ in range(size)]
        dist.all_gather(tensor_list, input)
        gathered_tensor = torch.cat(tensor_list, dim=1)
        return gathered_tensor
        
        


    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        
        rank, size = ctx.saved_tensors

        bs = (grad_output.shape[1] + size - 1)//size
        start = bs * rank
        trimmed_grad = grad_output[:, start:start+bs, ...]


        return trimmed_grad, None, None
    
class Gather1(torch.nn.Module):
    def __init__(self, rank, size):
        super().__init__()

        self.rank = rank
        self.size = size

    def forward(self, part_tensor):
        ans = gather1.apply(part_tensor, self.rank, self.size)
        return ans
    
class gather_grad1(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`

        return input
        
        


    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        
        
        dist.all_reduce(grad_output, dist.ReduceOp.SUM)

        return grad_output
    
class GatherGrad1(torch.nn.Module):
    def __init__(self, rank, size):
        super().__init__()

        self.rank = rank
        self.size = size

    def forward(self, tensor):
        ans = gather_grad1.apply(tensor)
        return ans