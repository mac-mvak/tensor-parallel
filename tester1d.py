import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn
import time
from gathers import Gather1, GatherGrad1
import random


x = torch.tensor([[1,2,3,4,5]], requires_grad=True, dtype=float)

y1 = torch.cos(x)
y2 = torch.softmax(x, dim=1)

print(y1)
print(y2)

z = (y1*y2).sum()
print(z)
z.backward()
print(x.grad)
