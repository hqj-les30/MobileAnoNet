import torch.utils.data as td
from utils import *
from multiprocessing import Pool
import torch
import numpy as np
from random import sample
import os
import csv
import re

class Oneclip(td.Dataset):
    def __init__(self, x: np.ndarray, y, d, n_frames = 64, hop = 1, random = None) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.n = n_frames
        if random is not None:
            self.inds = sample(list(range(0, x.shape[2]-n_frames)), k=random)
        else:
            self.inds = list(range(0, x.shape[2]-n_frames, hop))
        self.l = len(self.inds)
        self.d = d
    def __len__(self) -> int:
        return self.l
    def __getitem__(self, index):
        frame = self.x[:, :, self.inds[index]: self.inds[index]+self.n]
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        return frame, self.y, self.d

@torch.no_grad()
def Read_datas(filename, section, domain, hop, random):
    # data = loadPickle(filename)
    # x = data['x']
    # y = data['y']
    x = np.load(filename)
    dataset = Oneclip(x,int(section),domain,hop=hop,random=random)
    return dataset

def Readcsv(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

# def get_section(fname):
#     pattern = re.compile(r'(?<=section_)[0-9]+')
#     num = pattern.findall(fname)
#     if num is None:
#         return num
#     else: 
#         return int(num[0])

def Get_dataset(filelist, sections, domains, hop = 1, random = False):
    assert len(filelist) == len(sections)
    pool = Pool(60)
    datasets = pool.starmap(Read_datas, zip(filelist, sections, domains, [hop for _ in range(len(filelist))], [random for _ in range(len(filelist))]))
    pool.close()
    pool.join()
    dataset = td.ConcatDataset(datasets)
    return dataset

MT_LIST22 = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

MT_CODE22 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6
}

ATTRI_CODE22 = {
    'bearing': {
        'vel': {'6': 0, '10': 1, '14': 2, '18': 3, '22': 4, '2': 5, '16': 6, '24': 7, '12': 8, '20': 9, '26': 10, '8': 11, '4': 12},
        'loc': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7},
        'f-n': {'A': 0, 'B': 1, 'C': 2}
    },
    'fan': {
        'm-n': {'W': 0, 'X': 1, 'Y': 2, 'Z': 3},
        'f-n': {'A': 0, 'B': 1, 'C': 2},
        'n-lv': {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3}
    },
    'gearbox': {
        'volt': {'0': 0, '1': 1, '2': 2, '3': 3},
        'wt': {'0': 0, '100': 1, '150': 2, '200': 3, '250': 4, '30': 5, '80': 6, '130': 7, '180': 8, '230': 9, '50': 10},
        'id': {'05': 0, '08': 1, '13': 2, '00': 3, '02': 4, '11': 5}
    },
    'slider': {
        'vel': {'300': 0, '400': 1, '500': 2, '600': 3, '700': 4, '800': 5, '900': 6, '1000': 7, '1100': 8},
        'ac': {'0.03': 0, '0.04': 1, '0.05': 2, '0.06': 3, '0.07': 4, '0.08': 5, '0.09': 6, '0.10': 7, '0.11': 8},
        'f-n': {'A': 0, 'B': 1, 'C': 2}
    },
    'ToyCar': {
        'car': {'A1': 0, 'A2': 1, 'C1': 2, 'C2': 3, 'E1': 4, 'E2': 5},
        'speed': {'40V': 0, '28V': 1, '31V': 2, '37V': 3},
        'mic': {'1': 0, '2': 1}
    },
    'ToyTrain': {
        'car': {'A1': 0, 'A2': 1, 'C1': 2, 'C2': 3, 'E1': 4, 'E2': 5},
        'speed': {'6': 0, '7': 1, '9': 2, '10': 3},
        'mic': {'1': 0, '2': 1}
    },
    'valve': {
        'pat': {'0': 0, '1': 1, '2': 2, '3': 3},
        'panel': {'b-c': 0, 's-c': 1, 'bs-c': 2, 'open': 3},
        'v1pat': {'4': 0, '5': 1},
        'v2pat': {'4': 0, '5': 1},
    }
}

def find_files(filelist, key):
    str_match = [s for s in filelist if key in s]
    return str_match

class dcase22(td.Dataset):
    def __init__(self, machines: str, key: str, label1 = "section", label2 = "sou/tar", randomframe = None) -> None:
        super().__init__()
        datapath = "DCASE_data/feature/DCASE22"
        if machines == "all":
            self.machines = MT_LIST22
        else:
            self.machines = machines.split(',')
        # print(self.machines)
        self.machines = sorted(self.machines, key=lambda m: MT_CODE22[m])
        filenames = []
        self._machine_ids = []
        # self.section_ids = []
        self._domains = []
        self._flags = []
        self._labels = []
        self.attributes = []
        for m in self.machines:
            path = os.path.join(datapath, m)
            flist = [os.path.join(path, f) for f in os.listdir(path)]
            flist = find_files(flist, key=key)
            filenames.extend(flist)
            # machine_ids.extend([machines.index(m) for _ in flist])
            self._machine_ids.extend([MT_CODE22[m] for _ in flist])
            self._domains.extend([int("target" in f) for f in flist])
            self._flags.extend([int("anomaly" in f) for f in flist])
            self._labels.extend([get_section(f) + self.machines.index(m) * 3 for f in flist])
            print(f"classnum: {np.unique(self._labels).shape[0]}, max_label: {np.max(self._labels)}")

            if label2 == "sou/tar":
                self.attributes.extend([int("target" in f) for f in flist])
            elif label2 == "attri":
                self.attributes.extend(self.read_attri(m, flist))
            
            self.dataset = Get_dataset(filenames, self._labels, self.attributes, random = randomframe)
            
    def read_attri(self, m, flist):
        attripath = "DCASE_data/attributes_22"
        attripath = os.path.join(attripath, f"{m}_att.csv")
        c = r"section_([0-9]+)_(source|target)_(train|test)_(normal|anomaly)_([0-9]+)"
        rows = Readcsv(attripath)
        max_attr = 3
        list_attr = []
        for fname in flist:
            section, domain, setname, flag, id = re.findall(c, fname)[0]
            a = f"({m}/{setname}/)?section_{section}_{domain}_{setname}_{flag}_{id}_"
            r = list(filter(lambda x: re.match(a, x[0]) != None, rows)) # find the matching one\
            try:
                assert len(r) == 1   
            except:
                print(a)
                print(r)
            r = r[0]
            attributes = r[2::2] # get all the attributes
            attrinames = r[1::2] # get all the attrinames
            try:
                attributes = [ATTRI_CODE22[m][attrinames[i]][attributes[i]] for i in range(len(attrinames))] # map the attributes to labels
            except:
                print(r, attributes, attrinames)
            attributes.extend(-1 for _ in range(max_attr - len(attributes))) # pad to same length
            attributes = tuple(attributes)
            list_attr.append(attributes)

        return list_attr
            
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
                
    def clip_size(self) -> int:
        return len(self) // len(self._machine_ids)
    
    def n_class(self) -> int:
        return np.unique(self._labels).shape[0]
    
    @property
    def flags(self) -> np.ndarray:
        return np.array(self._flags)
    @property
    def domains(self) -> np.ndarray:
        return np.array(self._domains)
    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)
    @property
    def machine_ids(self) -> np.ndarray:
        return np.array(self._machine_ids)
    
MT_CODE = {'bearing': 0, 'fan': 1, 'gearbox': 2, 'slider': 3,
           'ToyCar': 4, 'ToyTrain': 5, 'valve': 6, 'bandsaw': 7, 'grinder': 8,
           'shaker': 9, 'ToyDrone': 10, 'ToyNscale': 11, 'ToyTank': 12, 'Vacuum': 13
           }

CLASS_ATTRI_ALL = {'bearing': ['vel', 'loc'],
                   'fan': ['m-n'],
                   'gearbox': ['volt', 'wt'],
                   'slider': ['vel', 'ac'],
                   'ToyCar': ['car', 'spd'],
                   'ToyTrain': ['car', 'spd'],
                   'valve': ['pat'],
                   'bandsaw': ['vel'],
                   'grinder': ['grindstone', 'plate'],
                   'shaker': ['speed'],
                   'ToyDrone': ['car', 'spd'],
                   'ToyNscale': ['car', 'spd'], 
                   'Vacuum': ['car', 'spd'],
                   'ToyTank': ['car', 'spd']
                   }

ATTRI_CODE = {'bearing': {'vel': {'5': 0, '7': 1, '8': 2, '9': 3, '11': 4, '13': 5, '15': 6, '16': 7, '17': 8, '19': 9, '21': 10},
                          'loc': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}},
              'fan': {'m-n': {'W': 0, 'X': 1, 'Y': 2, 'Z': 3}},
              'gearbox': {'volt': {'1.0': 0, '1.3': 1, '1.5': 2, '1.8': 3, '2.0': 4, '2.3': 5, '2.5': 6, '2.8': 7, '3.0': 8},
                          'wt': {'0': 0, '30': 1, '50': 2, '80': 3, '100': 4, '130': 5, '150': 6, '180': 7, '200': 8, '230': 9, '250': 10}},
              'slider': {'vel': {'300': 0, '380': 1, '400': 2, '440': 3, '490': 4, '500': 5, '540': 6, '580': 7, '600': 8, '620': 9, '660': 10, '700': 11, '730': 12, '800': 13, '900': 14, '1000': 15, '1100': 16},
                         'ac': {'0.03': 0, '0.04': 1, '0.05': 2, '0.06': 3, '0.07': 4, '0.08': 5, '0.09': 6, '0.10': 7, '0.11': 8, '0.30': 9}},
              'ToyCar': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5, 'D1': 6, 'D2': 7, 'E1': 8, 'E2': 9},
                         'spd': {'28V': 0, '31V': 1, '34V': 2, '37V': 3, '40V': 4},
                         'mic': {'1': 0, '2': 1}},
              'ToyTrain': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5, 'D1': 6, 'D2': 7, 'E1': 8, 'E2': 9},
                           'spd': {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4},
                           'mic': {'1': 0, '2': 1}},
              'valve': {'pat': {'00': 0, '01': 1, '02': 2, '03': 3}},
              'Vacuum': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5},
                         'spd': {'M': 0, 'H': 1, 'Lb': 2, 'Mb': 3}},
              'bandsaw': {'vel': {'13': 0, '15': 1, '12': 2, '14': 3}},
              'grinder': {'grindstone': {'1': 0, '2': 1, '3': 2, '4': 3},
                          'plate': {'1': 0, '2': 1, '3': 2}},
              'shaker': {'speed': {'1000': 0, '1500': 1, '1200': 2}},
              'ToyDrone': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5},
                           'spd': {'2': 0, '3': 1, '1': 2}},
              'ToyTank': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5},
                          'spd': {'1': 0, '3': 1, '4': 2, '5': 3}},
              'ToyNscale': {'car': {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5},
                            'spd': {'2': 0, '3': 1, '1': 2, '5': 3}}
              }

class dcase23(td.Dataset):
    def __init__(self, machines: str, key: str, label1 = "section", label2 = "sou/tar", randomframe = None) -> None:
        super().__init__()
        datapath = "DCASE_data/feature/DCASE23"
        if machines == "all":
            self.machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve',
                             'bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
        else:
            self.machines = machines.split(',')

        self.machines = sorted(self.machines, key=lambda m: MT_CODE[m])
        filenames = []
        self._machine_ids = []
        self._domains = []
        self._flags = []
        self._labels = []
        self.attributes = []
        for m in self.machines:
            path = os.path.join(datapath, m)
            flist = [os.path.join(path, f) for f in os.listdir(path)]
            flist = find_files(flist, key=key)
            filenames.extend(flist)
            self._machine_ids.extend([MT_CODE[m] for _ in flist])
            self._domains.extend([int("target" in f) for f in flist])
            self._flags.extend([int("anomaly" in f) for f in flist])
            self._labels.extend([self.machines.index(m) for _ in flist])

            if label2 == "sou/tar":
                self.attributes.extend([int("target" in f) for f in flist])
            elif label2 == "attri":
                self.attributes.extend(self.read_attri(m, flist))
            
            self.dataset = Get_dataset(filenames, self._labels, self.attributes, random = randomframe)
            
    def read_attri(self, m, flist):
        path = os.path.join("DCASE_data/attributes_23", m, "attributes_00.csv")
        c = r"(source|target)_(train|test)_(normal|anomaly)_([0-9]+)"
        rows = Readcsv(path)
        rows = rows[1::]
        attr_names = CLASS_ATTRI_ALL[m]
        max_attr = 3
        list_attr = []
        for fname in flist:
            domain, setname, flag, id = re.findall(c, fname)[0]
            a = f"({m}/{setname}/)?section_00_{domain}_{setname}_{flag}_{id}_"
            r = list(filter(lambda x: re.match(a, x[0]) != None, rows)) # find the matching one\
            try:
                assert len(r) == 1
            except:
                print(a)
                print(r)
            r = r[0]
            attributes = r[2::2] # get all the attributes
            attributes = [ATTRI_CODE[m][attr_names[i]][attributes[i]] for i in range(len(attr_names))] # map the attributes to labels
            attributes.extend(-1 for _ in range(max_attr - len(attributes))) # pad to same length
            attributes = tuple(attributes)
            list_attr.append(attributes)
        
        return list_attr
            
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
                
    def clip_size(self) -> int:
        return len(self) // len(self._machine_ids)
    
    def n_class(self) -> int:
        return np.unique(self._labels).shape[0]
    
    @property
    def flags(self) -> np.ndarray:
        return np.array(self._flags)
    @property
    def domains(self) -> np.ndarray:
        return np.array(self._domains)
    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)
    @property
    def machine_ids(self) -> np.ndarray:
        return np.array(self._machine_ids)
    
    
