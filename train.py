import os
import sys
import argparse
import importlib
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from projects import BaseExp
from utils import AvgMeter
from utils import synchronize, all_reduce_mean

def get_arg_parser():
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("exp_file", type=str, help="path to config file")
    parser.add_argument("-d", "--devices", default="0", type=str, help="device for training")
    parser.add_argument(
        "-eval",
        "--eval",
        action="store_true",
        help="enable evaluation during training, notice that self-supervised model are not supported evaluation",
    )
    parser.add_argument("--dist-backend", default="gloo", type=str, help="distributed backend")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    args = parser.parse_args()
    return args

def main(args:dict, rank:int, exp_args:dict):
    # -------------------------------- init cuda enviroment ------------------------- #
    dist.init_process_group(
        backend=args.dist_backend,
        rank=rank,
        world_size=args.world_size,
        init_method="env://?use_libuv=False"
    )
    torch.cuda.set_device(rank)
    synchronize()

    # -------------------------------- init exp enviroment ------------------------- #
    sys.path.insert(0, os.path.dirname(args.exp_file))
    exp_env = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
    exp:BaseExp = exp_env.Exp(exp_args)
    logger = exp.logger
    tb_writer = exp.tb_writer

    if rank == 0:
        args_list = ["------------ Exp args -------------"]
        for key in exp.__dict__:
            if isinstance(exp.__dict__[key], (str, int, float)):
                args_list.append("{}: {}".format(key, exp.__dict__[key]))
        logger.info("\n".join(args_list)+"\n-------------------------")

    model = exp.get_model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    train_loader = exp.get_data_loader(train=True)
    if args.eval:
        eval_loader = exp.get_data_loader(train=False)
    optimizer = exp.get_optimizer()
    lr_scheduler = exp.get_lr_scheduler()

    if args.amp:
        if rank == 0:
            logger.info("Automatic Mixed Precision is enabled!")
        scaler = GradScaler()

    # ------------------------------------- train ------------------------------ #
    if rank == 0:
        logger.info("Training start...")
    ITERS_PER_EPOCH = len(train_loader)
    for epoch in range(exp.last_epoch, exp.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            data_iter = tqdm(enumerate(train_loader),
                            desc="EP_train_{}".format(epoch),
                            total=len(train_loader),
                            bar_format="{l_bar}{r_bar}")
        else:
            data_iter = enumerate(train_loader)

        model.train()
        avg_loss = AvgMeter()
        for i, (inputs, target) in data_iter:
            inputs, target = exp.data_preprocess(inputs, target)
            iter_count = epoch * ITERS_PER_EPOCH + i
            
            optimizer.zero_grad()
            if args.amp:
                with autocast('cuda'):
                    output = model(inputs)
                    loss = exp.calc_loss(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(inputs)
                loss = exp.calc_loss(output, target)
                loss.backward()
                optimizer.step()
            lr_scheduler.step(iter_count)
            
            tb_writer.add_scalar("Loss", loss.item(), iter_count)
            avg_loss.update(loss.item())
            if (i+1) % (ITERS_PER_EPOCH // 20) == 0 and rank == 0:
                log_str = "epoch: {}, iteration: {}/{}, lr: {}, loss: {}".format(
                    epoch, i, ITERS_PER_EPOCH, optimizer.param_groups[0]['lr'], loss.item())
                logger.info(log_str)
        log_str = "Train epoch: {}, loss: {}".format(epoch, avg_loss.avg)
        if rank == 0:
            logger.info(log_str)

        # ------------------------------------- eval ------------------------------ #
        synchronize()
        model.eval()
        metric_avg:dict[str, AvgMeter] = {}
        if args.eval:
            if rank == 0:
                data_iter = tqdm(enumerate(eval_loader),
                                desc="EP_eval_{}".format(epoch),
                                total=len(eval_loader),
                                bar_format="{l_bar}{r_bar}")
            else:
                data_iter = enumerate(eval_loader)
            with torch.no_grad():
                for i, (inputs, target) in data_iter:
                    if args.amp:
                        with autocast('cuda'):
                            output = model(inputs)
                            eval_result = exp.run_eval(output, target)
                    else:
                        output = model(inputs)
                        eval_result = exp.run_eval(output, target)
                    for k, v in eval_result.items():
                        v = all_reduce_mean(v)
                        if k not in metric_avg.keys():
                            metric_avg[k] = AvgMeter()
                        metric_avg[k].update(v.item())
                for k, avgmeter in metric_avg.items():
                    tb_writer.add_scalar("Eval/{}".format(k), avgmeter.avg, epoch)
                if rank == 0:
                    log_str = "Val epoch: {}, {}".format(
                        epoch, ", ".join(["{}: {:.4f}".format(k, avgmeter.avg) for k, avgmeter in metric_avg.items()]))
                    logger.info(log_str)

        # ------------------------------------- dump weights ------------------------------ #
        synchronize()
        if rank == 0  and (epoch % 2 == 0 or epoch == exp.max_epoch - 1):
            ckpt = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": exp.last_epoch
            }
            for k, avgmeter in metric_avg.items():
                ckpt[k] = avgmeter.avg
            torch.save(ckpt, "exp/{}/checkpoint/{}.pth".format(exp.save_folder_name, epoch))
        exp.last_epoch += 1

def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids

if __name__ == '__main__':
    args = get_arg_parser()
    exp_args = dict(
        # lr = 1e-3
    )

    # -------------------------------- init cuda dist ------------------------- #
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11451'
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = len(args.devices.split(","))
    args.world_size = nr_gpu

    # ------------------------------ Start dist training ---------------------- #
    processes = []
    for rank in range(nr_gpu):
        p = mp.Process(target=main, args=(args, rank, exp_args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()