import torch
import torch.distributed as dist
import torch.nn.functional as F



def train_loop(rank, size, model, epoch, optimizer, device, trainloader, logger):
    if rank == 0:
        lens =  torch.empty(1, dtype=int, device=device)
        lens[0] = len(trainloader)

        scatter_list = [lens.clone() for _ in range(size)]
        lens = torch.empty(1, dtype=int, device=device)
        dist.scatter(lens, scatter_list=scatter_list, src=0)
    else:
        lens = torch.empty(1, dtype=int, device=device)
        dist.scatter(lens, src=0)
    
    len_train = lens[0].item()

    model.train()
    running_loss = 0.0
    if rank == 0:
        dataiter = iter(trainloader)
    for i in range(len_train):
        # get the inputs; data is a list of [inputs, labels]
        if rank == 0:
            inputs, labels = next(dataiter)
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = torch.tensor(inputs.shape[0], dtype=int, device=device)
            scatter_list = [batch_size.clone() for _ in range(size)]
            batch_size = torch.tensor(1, dtype=int, device=device)
            dist.scatter(batch_size, scatter_list=scatter_list, src=0)
            scatter_list = [labels for _ in range(size)]
            dummy = torch.empty_like(labels)
            dist.scatter(dummy, scatter_list=scatter_list, src=0)
            scatter_list = [inputs for _ in range(size)]
            dummy = torch.empty_like(inputs)
            dist.scatter(dummy, scatter_list=scatter_list, src=0)
        else:
            batch_size = torch.tensor(1, dtype=int, device=device)
            dist.scatter(batch_size, src=0)
            labels = torch.empty(batch_size, dtype=int, device=device)
            dist.scatter(labels, src=0)
            inputs = torch.empty((batch_size, 3, 32, 32), device=device) # тут для случая Cifar иначе, можно передать другое без особых проблем.
            dist.scatter(inputs, src=0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.detach().item()
        if i % 100 == 99 and rank == size-1:    # print every 2000 mini-batches
            print_loss = running_loss / len_train
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {print_loss:.3f}')
            logger.log({'loss': print_loss}, step= epoch * len_train * inputs.shape[0] + i * batch_size)
            
            running_loss = 0.0
    dist.barrier()



@torch.no_grad()
def test_loop(rank, size, model, epoch, device, testloader, logger):
    if rank == 0:
        lens =  torch.empty(1, dtype=int, device=device)
        lens[0] = len(testloader)

        scatter_list = [lens.clone() for _ in range(size)]
        lens = torch.empty(1, dtype=int, device=device)
        dist.scatter(lens, scatter_list=scatter_list, src=0)
    else:
        lens = torch.empty(1, dtype=int, device=device)
        dist.scatter(lens, src=0)
    
    len_test = lens[0].item()

    model.eval()
    if rank == 0:
        dataiter = iter(testloader)
    elif rank == size - 1:
        correct = 0
        total = 0
    for i in range(len_test):
        # get the inputs; data is a list of [inputs, labels]
        if rank == 0:
            inputs, labels = next(dataiter)
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = torch.tensor(inputs.shape[0], dtype=int, device=device)
            scatter_list = [batch_size.clone() for _ in range(size)]
            batch_size = torch.tensor(1, dtype=int, device=device)
            dist.scatter(batch_size, scatter_list=scatter_list, src=0)
            scatter_list = [labels for _ in range(size)]
            dummy = torch.empty_like(labels)
            dist.scatter(dummy, scatter_list=scatter_list, src=0)
            scatter_list = [inputs for _ in range(size)]
            dummy = torch.empty_like(inputs)
            dist.scatter(dummy, scatter_list=scatter_list, src=0)
        else:
            batch_size = torch.tensor(1, dtype=int, device=device)
            dist.scatter(batch_size, src=0)
            labels = torch.empty(batch_size, dtype=int, device=device)
            dist.scatter(labels, src=0)
            inputs = torch.empty((batch_size, 3, 32, 32), device=device) # тут для случая Cifar иначе, можно передать другое без особых проблем.
            dist.scatter(inputs, src=0)

        outputs = model(inputs)
        if rank == size - 1:
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    if rank == size - 1:
        print(f'Accuracy of the network on the train images on {epoch+1}: {100 * correct // total} %')
        logger.log({'accuracy': correct / total, 'epoch': epoch})

        # print statistics
    
    dist.barrier()

