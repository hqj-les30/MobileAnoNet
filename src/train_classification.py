import os
import argparse
import logging
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as td
from models import build_model
from scipy.stats import hmean
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
from datasets import dcase22, dcase23


def worker(rank, world_size, args):
    device_ids = args.device_ids.split(',')
    device = torch.device(f'cuda:{device_ids[rank]}')

    logger = logging.getLogger("train-logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(args.modelpath, "log.txt"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s---    %(message)s    ---%(funcName)s"))
    logger.addHandler(handler)
    if rank == 0:
        print('loading data...')
        logger.info(f"start training with {args.mt_train}, dataset: {args.data}, random seed: {args.seed}")
        
    if args.data == "DCASE22":
        Trainset = dcase22(machines=args.mt_train, key = 'train', label2="attri", randomframe=20)
    elif args.data == "DCASE23":
        Trainset = dcase23(machines=args.mt_train, key = 'train', label2="attri", randomframe=20)

    net = build_model(in_channels=1, n_class=Trainset.n_class(), t=args.data)

    if args.resume is not None:
        ckpt = torch.load(args.resume)
        new_state_dict = {}
        for k,v in ckpt.items():
            new_state_dict[k[7:]] = v
        net.load_state_dict(new_state_dict)

    net.to(device)

    if len(device_ids) > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device=device)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[device], output_device=device, find_unused_parameters=True)

        train_sampler = td.DistributedSampler(Trainset)
        train_loader = td.DataLoader(
            Trainset,
            batch_size=args.bs,
            sampler=train_sampler,
            num_workers=2,
            drop_last=True
        )

    else:
        torch.cuda.set_device(device=device)
        train_loader = td.DataLoader(
            Trainset,
            batch_size=args.bs,
            num_workers=5
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.test_interval, gamma=0.8)

    for i in range(args.nepoch):
        if len(device_ids) > 1:
            train_loader.sampler.set_epoch(i)
        if rank == 0:
            print(f"epoch_{i+1}")

        train_loss = train_one_epoch(net, device, train_loader, optimizer)
        scheduler.step()
        if rank == 0:
            torch.save(net.state_dict(), os.path.join(args.modelpath, f"{datecode()}{args.marker}_epoch_{i+1}.pth"))
            logger.info(f"epoch_{i+1}: startloss: {round(train_loss[0], 4)}, endloss: {round(train_loss[-1], 4)}, maxloss: {round(train_loss.max(), 4)}, minloss: {round(train_loss.min(), 4)}")

        if (i + 1) % args.test_interval == 0:
            machines = args.mt_test
            if machines == "all":
                machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
            else:
                machines = machines.split(',')
            machine_rank = machines[rank::world_size]
            
            for m in machine_rank:
                print(f"testing for {m} ...")
                if args.data == "DCASE23":
                    trainset_m = dcase23(machines=m, key="test")
                    testset_m = dcase23(machines=m, key="test")
                elif args.data == "DCASE22":
                    trainset_m = dcase22(machines=m, key="test")
                    testset_m = dcase22(machines=m, key="test")

                dict_metrics = test_model(
                    net=net,
                    device=device,
                    Testset=testset_m,
                    Trainset=trainset_m
                )
                # da_methods = sorted(dict_metrics.keys(), key=lambda k: hmean(dict_metrics[k]))
                # best_method = da_methods[-1]
                for k in dict_metrics.keys():
                    logger.info(f"machine [{m}], AD_method[{k}], aucs: {dict_metrics[k][1]}, auct: {dict_metrics[k][0]}, pauc: {dict_metrics[k][2]}, hmean: {round(hmean(dict_metrics[k]), 4)}")
            
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str, default="6", help="the device to run the model")
    parser.add_argument("--mt_train", type=str, default="all", help="which machine to train")
    parser.add_argument("--mt_test", type=str, default="all", help="which machine to test")
    parser.add_argument("--data", type=str, default="DCASE23", help="dataset to run")
    parser.add_argument("--modelpath", type=str, default="experiments/exp", help="path to save result")
    parser.add_argument("--resume", type=str, default=None, help="path to load pretrained model")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--nepoch", type=int, default=1, help="nepoch")
    parser.add_argument("--bs", type=int, default=160, help="batch_size per gpu")
    parser.add_argument("--test_interval", type=int, default=1, help="test per n epochs")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--marker", type=str, default="", help="special name of one experiment")
    parser.add_argument("--port", type=str, default="5679", help="port to run DDP")
    args = parser.parse_args() 
    
    device_ids = args.device_ids.split(',')
    
    world_size = len(device_ids)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    if args.seed is not None:
        setup_seed(args.seed)

    mp.spawn(worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
            )

if __name__ == '__main__':
    main()