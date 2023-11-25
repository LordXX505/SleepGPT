
import argparse
import os
# from SpindleDetection.utils.other import By_Event
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import time

from torch.distributed.elastic.multiprocessing.errors import record
# from SpindleDetection.utils.other import Logger
def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/gpfs/share/home/2201210064/vit/checkpoint").is_dir():
        p = Path(f"/gpfs/share/home/2201210064/vit/checkpoint/experiment")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

@record
def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--init_method', type=str)
    parse.add_argument('--rank', type=int)
    parse.add_argument('--ws', type=int)
    parse.add_argument('--gpu', type=int)
    parse.add_argument('--world_size', type=int)
    args = parse.parse_args()

    # args.job_dir = get_shared_folder() / "%j"
    # args.dist_url = get_init_file().as_uri()

    # if 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # os.environ['LOCAL_RANK'] = str(args.gpu)
    # os.environ['RANK'] = str(args.rank)
    # os.environ['WORLD_SIZE'] = "4"    #str(args.world_size)
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ["LOCAL_RANK"])
    # os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
    # os.environ['MASTER_PORT'] = '12345'
    # os.environ['MASTER_ADDR'] = 'gpu11'
    print('./result/temp%s.txt' % args.gpu)
    print(os.environ)
    print(torch.cuda.nccl.version())
    print(torch.cuda.device_count())
    a = (0, 1, 2, 3)
    print(a,  args.gpu)
    args.init_method = 'ENV'
    args.distributed = True
    args.dist_backend = 'nccl'
    if args.init_method == 'TCP':
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

        print('| distributed init (rank {}): {}, gpu {}, backend{}'.format(
            args.rank, args.dist_url, args.gpu, args.dist_backend), flush=True)
        # torch.distributed.init_process_group(backend=args.dist_backend)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        print("finish init_process_group rank{} gpu{}".format(args.rank, args.gpu), flush=True)
    elif args.init_method == 'ENV':
        dist.init_process_group('nccl', init_method='env://')
    else:
        print('| distributed init (rank {}): {}, gpu {}, backend {}'.format(
            args.rank, args.dist_url, args.gpu, args.dist_backend), flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                              world_size=args.world_size, rank=args.rank)

    print('barrier')
    torch.cuda.set_device(args.gpu)

    # t = torch.tensor(1).to(int(os.environ['LOCAL_RANK']))
    # dist.all_reduce(t)
    # print(t)
    rank = dist.get_rank()
    if rank == 0:
        print(f"rank = {rank} is initialized")
        output = torch.ones((128, 5120)) * torch.abs(torch.randn((128, 5120)))
        output = output.to('cuda')
        print(output)

        target = torch.rand((128, 5120)).to('cuda')
        target = torch.ones_like(target) * (target > 0.2)
        print(target)
        # By_Event_model = By_Event(0.55, 0.2, freq=20, time=1, device='cuda')
        print('start:{}_{}'.format(time.time(), args.gpu))

        res = output-target
        print('end:{}_{}'.format(time.time(), args.gpu))

        print(res)
        dist.barrier()
    else:
        output = torch.ones((128, 5120)) * torch.abs(torch.randn((128, 5120)))
        output = output.to('cuda')
        print(output)

        target = torch.rand((128, 5120)).to('cuda')
        target = torch.ones_like(target) * (target > 0.2)
        print(target)
        #
        # By_Event_model = By_Event(0.55, 0.2, freq=20, time=1,  device='cuda')
        print('start:{}_{}'.format(time.time(), args.gpu))

        res = output - target

        print('end:{}_{}'.format(time.time(), args.gpu))
        print(res)
        dist.barrier()
    res = torch.tensor(res, device='cuda')
    print(res)
    dist.barrier()
    print('all_reduce')
    dist.all_reduce(res)
    print('all_reduce_done')
    print(res/args.world_size)
    print(args.world_size)


if __name__ =='__main__':
    main()