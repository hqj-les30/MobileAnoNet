import argparse
import logging
import numpy as np
import torch
import os
import multiprocessing as mp
import tqdm
import random
import re
from utils import *

d22 = {
    'datapath': "DCASE_data/origin/DCASE22",
    'targetpath': "DCASE_data/feature/DCASE22",
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
}

d23 = {
    'datapath': "DCASE_data/origin/DCASE23",
    'targetpath': "DCASE_data/feature/DCASE23",
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': ['bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
}

def file_trans(filename, targetpath, nffts, nframes):
    a = r"section_([0-9]+)_(source|target)_(train|test)_(normal|anomaly)_([0-9]+)"
    section, domain, setname, flag, id = re.findall(a, filename)[0]

    x = STFT_spectrum(f_name=filename, n_ffts=nffts)
    x = x.astype(np.float32)
    
    # save data
    path = os.path.join(targetpath, f"section_{section}_{domain}_{setname}_{flag}_{id}.npy")
    np.save(path, x)
    return path


def Transform(path_set, machine, args, d):
    path_train = os.path.join(path_set, machine, "train")
    path_test = os.path.join(path_set, machine, "test")
    path_tar = os.path.join(d['targetpath'], machine)
    
    if not os.path.isdir(path_tar):
        os.mkdir(path_tar)
        
    flist = [os.path.join(path_train, f) for f in os.listdir(path_train) if '.Identifier' not in f]
    if os.path.isdir(path_test):
        flist += [os.path.join(path_test, f) for f in os.listdir(path_test)  if '.Identifier' not in f]
        
    
    pool = mp.Pool()
    pool.starmap(file_trans, [(f, path_tar, args.nffts, args.nframes) for f in flist])
    pool.close()
    pool.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="DCASE22", help="dataset")
    parser.add_argument("--nffts", type=int, default="2048", help="points for FFT")
    parser.add_argument("--nframes", type=int, default="192", help="length per frame")

    args = parser.parse_args()
    
    if args.set == "DCASE22":
        d = d22
    else:
        d = d23

    path_set = os.path.join(d['datapath'], "dev_data")

    for machine in d['dev']:
        print(f"transforming dev features for {machine} ...")
        Transform(path_set, machine, args, d)
        
    path_set = os.path.join(d['datapath'], "eval_data")

    for machine in d['eval']:
        print(f"transforming add features for {machine} ...")
        Transform(path_set, machine, args, d)

if __name__ == '__main__':
    main()
