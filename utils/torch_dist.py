import torch
from torch import distributed as dist

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        if not isinstance(x, torch.Tensor):
            x_reduce = torch.tensor(x).cuda()
            dist.all_reduce(x_reduce)
            x_reduce /= world_size
            return x_reduce
        else:
            x_reduce = x.clone().cuda()
            dist.all_reduce(x_reduce)
            x_reduce /= world_size
            return x_reduce
    else:
        return torch.tensor(x).cuda()

def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()