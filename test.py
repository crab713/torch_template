import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '11451'

dist.init_process_group("gloo", rank=0, world_size=1, init_method="env://?use_libuv=False")
print(dist.get_world_size())

import os
print(os.getcwd())