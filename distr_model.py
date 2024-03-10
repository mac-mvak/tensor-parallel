import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import Process
import random
import argparse
import wandb

import os
from loops import train_loop, test_loop

from model_shards import ModelShard



def init_process(rank, size, batch_size, fn, master_port, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, batch_size)

def main_train(rank, size, batch_size):
    torch.random.manual_seed(0)
    if rank == 0:
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    else:
        testloader = trainloader = None

    device = torch.device(rank)
    model = ModelShard(rank, size)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    if rank == size - 1:
        logger = wandb.init(
            project="tensor_parallel",
            name=f'{size} process, bs={batch_size}'
        )
    else:
        logger = None
    for epoch in range(10):
        train_loop(rank, size, model, epoch, optimizer, device, trainloader, logger)
        test_loop(rank, size, model, epoch, device, testloader, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, help='batch_size', default=64)
    parser.add_argument('-s', type=int, help='size', default=4)
    args = parser.parse_args()
    size = args.s
    batch_size = args.b
    processes = []
    port = random.randint(25000, 30000)
    assert 100 % size == 0
    assert 32 % size == 0
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, batch_size, main_train, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
