import json
import os
import re
import numpy as np
import librosa
import tqdm
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean
import da
import torch
import torch.utils.data as td
import random
import datetime

DA = ['maha', 'knn', 'lof', 'smah', 'cos', 'scos']

def train_one_epoch(net, device, trainloader, optimizer):
    net.train()
    train_loss = []
    for x, y, z in tqdm.tqdm(trainloader):
        x, y, z = x.to(device), y.to(device), [r.to(device) for r in z]
        loss = net(x, y, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return np.array(train_loss)

def eval_one_ep(embs_train, train_istarget, train_section,
    embs_test, test_isano, test_istarget, test_section
    ):
    uniq_section = np.unique(test_section)
    dicts = None
    for sec in uniq_section:
        index_train = train_section == sec
        index_test = test_section == sec
        if dicts is None:
            dict_metrics = eval_one_machine(
                embs_train[index_train], train_istarget[index_train], embs_test[index_test],
                test_isano[index_test], test_istarget[index_test]
            )
            dicts = dict_metrics
        else:
            dict_metrics = eval_one_machine(
                embs_train[index_train], train_istarget[index_train], embs_test[index_test],
                test_isano[index_test], test_istarget[index_test]
            )
            dicts = update_dict(dicts, dict_metrics)
            
    return dicts

def update_dict(a: dict, b: dict):
    nd = {}
    for k, vb in b.items():
        va = a[k]
        n = [hmean([va[i], vb[i]]) for i in range(len(va))]
        nd[k] = n

    return nd

def eval_one_machine(embs_train, train_istarget, 
    embs_test, test_isano, test_istarget,
    das = DA
    ):

    dict_metrics = {}
    
    
    if 'maha' in das:
        scores, _ = da.maha_scores(embs_test=embs_test, embs_train=embs_train)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['maha'] = [auct, aucs, pauc]
        
    if 'knn' in das:
        scores, _ = da.knn_scores(embs_test=embs_test, embs_train=embs_train)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['knn'] = [auct, aucs, pauc]
        
    if 'lof' in das:
        scores, _ = da.lof_scores(embs_test=embs_test, embs_train=embs_train)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['lof'] = [auct, aucs, pauc]
        
    if 'cos' in das:
        scores, _ = da.cos_scores(embs_test=embs_test, embs_train=embs_train)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['cos'] = [auct, aucs, pauc]
        
    if 'smah' in das:
        scores_target, _ = da.maha_scores(embs_test=embs_test, embs_train=embs_train[train_istarget==1])
        scores_source, _ = da.maha_scores(embs_test=embs_test, embs_train=embs_train[train_istarget==0])
        scores = np.minimum(scores_target, scores_source)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['smah'] = [auct, aucs, pauc]
        
    if 'scos' in das:
        scores_target, _ = da.cos_scores(embs_test=embs_test, embs_train=embs_train[train_istarget==1])
        scores_source, _ = da.cos_scores(embs_test=embs_test, embs_train=embs_train[train_istarget==0])
        scores = np.minimum(scores_target, scores_source)
        auct, aucs, pauc = auc_pauc_per_sec(scores, test_isano, test_istarget)
        dict_metrics['scos'] = [auct, aucs, pauc]

    return dict_metrics

@torch.no_grad()
def test_model(net, device, Trainset, Testset):
    net.eval()
    test_loader = td.DataLoader(
        Testset,
        batch_size=Testset.clip_size(),
        shuffle=False,
        num_workers=5
    )

    train_loader = td.DataLoader(
        Trainset,
        batch_size=Trainset.clip_size(),
        shuffle=False,
        num_workers=5
    )

    embs_test = []

    for x, y, _ in test_loader:
        x, y = x.to(device), y.to(device)
        emb, _ = net(x)
        emb = emb.mean(0)
        embs_test.append(emb.cpu().numpy())

    embs_train = []
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        emb, _ = net(x)
        emb = emb.mean(0)
        embs_train.append(emb.cpu().numpy())

    embs_test, embs_train = np.stack(embs_test), np.stack(embs_train)
    
    dict_metrics = eval_one_ep(
        embs_test=embs_test, embs_train=embs_train, train_section=Trainset.labels,
        train_istarget=Trainset.domains, test_section=Testset.labels,
        test_isano=Testset.flags, test_istarget=Testset.domains,
    )

    return dict_metrics

def get_section(fname):
    pattern = re.compile(r'(?<=section_)[0-9]+')
    num = pattern.findall(fname)
    if num is None:
        return num
    else: 
        return int(num[0])
    
def get_epoch(fname):
    pattern = re.compile(r'(?<=epoch_)[0-9]+')
    num = pattern.findall(fname)
    if num is None:
        return num
    else: 
        return int(num[0])

def get_machine_name(fname, mas):
    pattern = re.compile(r'|'.join(mas))
    mname = pattern.findall(fname)
    return mname[0]

def auc_pauc_per_sec(scores, isano, istarget):
    aucs_index = (istarget==0) | (isano==1)
    aucs = roc_auc_score(
        isano[aucs_index],
        scores[aucs_index]
    )
    auct_index = (istarget==1) | (isano==1)
    auct = roc_auc_score(
        isano[auct_index],
        scores[auct_index]
    )
    pauc = roc_auc_score(
        isano, scores,
        max_fpr=0.1
    )
    
    aucs, auct, pauc = round(aucs, 3), round(auct, 3), round(pauc, 3)
    
    return auct, aucs, pauc

def find_files(filelist, key):
    str_match = [s for s in filelist if key in s]
    return str_match

def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("notfound")

def STFT_spectrum(f_name: str, n_ffts: int, n_channels: int = 1):
    y, _ = file_load(f_name)
    mag_spec = abs(librosa.stft(y, n_fft=n_ffts, hop_length=512))
    if n_channels == 1:
        mag_spec = mag_spec[None]
    return mag_spec
    
def find_int(string):
    pattern = re.compile(r'\d+')
    num = pattern.findall(string)
    if num is None:
        return num
    else: 
        return int(num[0])

def readcsv(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([int(d) for d in row])

    return rows

def Readcsv(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)

def datecode():
    return datetime.datetime.now().strftime('%m%d')
