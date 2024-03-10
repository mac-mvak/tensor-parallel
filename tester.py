import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn
import time
from gathers import Gather1, GatherGrad1
import random

def init_process(rank, size, fn, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    



def run(rank, size):
    """ Distributed function to be implemented later. """
    if rank==1:
        linear = torch.ones((1,5), requires_grad=True)
        gat_grad = GatherGrad1(rank, 2)
        gat = Gather1(rank, 2)
        linear = linear * 2

    if rank==0:
        linear = torch.ones((1, 5), requires_grad=True)
        gat_grad = GatherGrad1(rank, 2)
        gat = Gather1(rank, 2)



    x = torch.tensor([[1,2,3,4,5]], requires_grad=True, dtype=float)
    y = gat_grad(x)
    if rank==0:
        y = torch.cos(y)
    elif rank == 1:
        y = torch.softmax(y, dim=1)
    y = gat(y)
    print(rank, y.shape)
    y1 = y[0, :5]
    y2 = y[0, 5:]
    print(rank,  y1)
    z = (y[0, :5] * y[0, 5:]).sum()
    z.backward()
    #print(rank, x)
    print(rank, z)
    print(rank,  x.grad)
    

if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
