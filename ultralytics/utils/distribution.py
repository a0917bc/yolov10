import os
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if dist.is_available() and dist.is_initialized():
        return True
    return False

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('distributed init rank %d, gpu%d'%(args.rank, args.gpu))
    
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
    
def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1