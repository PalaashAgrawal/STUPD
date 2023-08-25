import torch
from torch.utils.data import Dataset
from .utils import stupd_classes, convert_stupdBbox_to_spatialSenseBbox
from .baseData import baseData

import json
from pathlib import Path
import pandas as pd

def noop(x): return x

class coordinate2D_OnlyDataset(baseData):
    def __init__(self,
                annotations_path, 
                x_tfms: list = None,
                y_tfms: list = None):
            
        super().__init__()

        self.coords = []
        self.predicates = [] #y
        
        self.classes = sorted(stupd_classes)
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        self.x_tfms = list(x_tfms or [noop]) 
        self.y_tfms = list(y_tfms or [noop]) + [lambda y: self.class2idx[y]]
        
        assert Path(annotations_path).exists()
        annotation_files = [o for o in annotations_path.iterdir() if str(o).endswith('csv') and o.stem in self.classes]

        for annotation in annotation_files:
            relations = pd.read_csv(annotation).dropna() #any row with incomplete data is dropped

            for k,row in relations.iterrows():

                self.predicates.append(row['relation'])                
                subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[0], eval(row['object_bbox2d'])[0]
                subj_position, obj_position = eval(row['subject_position3d'])[0], eval(row['object_position3d'])[0]

                self.coords.append([subj_position['x']-obj_position['x'],
                                    subj_position['y']-obj_position['y'],
                                    *convert_stupdBbox_to_spatialSenseBbox(subj_bbox),
                                    *convert_stupdBbox_to_spatialSenseBbox(obj_bbox),
                                    ])

        self.model = 'coordinateOnly'
    
    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):
        coord = torch.Tensor(self.apply_tfms(self.coords[i], self.x_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_tfms)])
    
        if torch.cuda.is_available(): coord, predicate = coord.type(torch.cuda.FloatTensor), predicate.type(torch.cuda.LongTensor)
        
        return (coord, predicate)
        

    