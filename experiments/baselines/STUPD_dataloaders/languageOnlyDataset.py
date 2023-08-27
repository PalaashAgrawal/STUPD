import torch
from .baseData import baseData

from .utils import stupd_classes, word2vec, convert_stupdBbox_to_spatialSenseBbox
import json
from pathlib import Path
from torch import nn
import pandas as pd



def noop(x): return x



class languageOnlyDataset(baseData):
    def __init__(self, 
                annotations_path,
                encoder_path,
                x_tfms: list = None,
                y_tfms: list = None):
        
        f'''
        annotations_directory_path: path of the directory containing json files for ViDVRD annotations. 
        encoder_path = path of word2vec encoder
        x_tfms: transforms that need to be applied on the subject and object words. NOTE: no need to include transforms to convert them into vectors. 
        '''

        super().__init__()

    
        self.subjects = [] #x1
        self.objects = [] #x2
        self.predicates = [] #y

        self.classes = sorted(stupd_classes)
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        # #transforms
        assert Path(encoder_path).exists(), f'Invalid path for word2vec encoder'

        self.x_tfms = list(x_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_tfms = list(y_tfms or [noop]) + [lambda y: self.class2idx[y]]
        


        # assert Path(annotations_directory_path).exists()
        # files = [f for f in Path(annotations_directory_path).iterdir() if str(f).endswith('json')]

        # for annotations_path in files:
        #     annotations = json.load(open(annotations_path))
        #     id2obj = self._obj_id2obj(annotations)
        #     for relation in annotations["relation_instances"]:
        #         predicate = self._get_valid_predicate(relation["predicate"], vidvrd_to_stupd)
        #         if predicate is None: continue

        #         self.predicates.append(predicate)
        #         self.subjects.append(id2obj[relation["subject_tid"]])
        #         self.objects.append(id2obj[relation["object_tid"]])
        
        
        assert Path(annotations_path).exists()
        annotation_files = [o for o in annotations_path.iterdir() if str(o).endswith('csv') and o.stem in self.classes]
        
        for annotation in annotation_files:
            relations = pd.read_csv(annotation).dropna()
            for _,row in relations.iterrows():
                
                self.predicates.append(row['relation'])
                self.subjects.append(f"{row['subject_category']} {row['subject_supercategory']}")
                self.objects.append(f"{row['object_category']} {row['object_supercategory']}")
                    
        self.model = 'languageOnly'
    
    def __len__(self): return len(self.predicates)
    
    def __getitem__(self, i):
        subj = self.apply_tfms(self.subjects[i], self.x_tfms)
        obj =  self.apply_tfms(self.objects[i] , self.x_tfms)
        predicate = self.apply_tfms(self.predicates[i], self.y_tfms)
        
        return (torch.Tensor(subj).type(torch.cuda.FloatTensor), 
                torch.Tensor(obj).type(torch.cuda.FloatTensor), 
                torch.Tensor([predicate]).type(torch.cuda.LongTensor))

    def apply_tfms(self, o, tfms):
        for tfm in tfms: o = tfm(o)
        return o


    