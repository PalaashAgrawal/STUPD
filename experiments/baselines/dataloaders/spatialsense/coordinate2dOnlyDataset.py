import torch
from torch.utils.data import Dataset
from .utils import spatialsenses_to_stupd
from .baseData import baseData

import json
from pathlib import Path

def noop(x): return x

class coordinate2D_OnlyDataset(baseData):
    def __init__(self,
                annotations_path, 
                split=None, 
                x_tfms: list = None,
                y_tfms: list = None):
            
        super().__init__()

        self.coords = []
        self.predicates = [] #y
        
        self.split = split
        if self.split is not None: assert split in ['train', 'valid', 'test'], f"invalid selection of split. expected values = 'train', 'valid', 'test'"
        
        self.classes = sorted(list(set(spatialsenses_to_stupd.values())))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        self.x_tfms = list(x_tfms or [noop]) 
        self.y_tfms = list(y_tfms or [noop]) + [lambda y: self.class2idx[y]]
        

        assert Path(annotations_path).exists()
        for relations in json.load(open(annotations_path)):
            if self.split and not relations["split"] == split: continue
            for relation in relations['annotations']:
                if not relation['label']: continue

                self.coords.append([relation['subject']['x'] - relation['object']['x'], 
                                    relation['subject']['y'] - relation['object']['y'], 
                                    *relation['subject']['bbox'], 
                                    *relation['object']['bbox']])
                 #self.coords.append([relation['subject']['x'], relation['subject']['y'], relation['object']['x'], relation['object']['x']])
                 #self.coords.append([relation['subject']['x'] - relation['object']['x'], relation['subject']['y'] - relation['object']['y']])
                self.predicates.append(relation['predicate'])

        self.model = 'coordinateOnly'
    
    def __len__(self): return len(self.coords)

    def __getitem__(self, i):
        coord = torch.Tensor(self.apply_tfms(self.coords[i], self.x_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_tfms)])


    
        if torch.cuda.is_available(): coord, predicate = coord.type(torch.cuda.FloatTensor), predicate.type(torch.cuda.LongTensor)
        
        return (coord, predicate)
        

    