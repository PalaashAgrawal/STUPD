import torch
from .baseData import baseData

from .utils import spatialsenses_to_stupd, word2vec
import json
from pathlib import Path
from torch import nn


def noop(x): return x



class languageOnlyDataset(baseData):
    def __init__(self, 
                annotations_path,
                encoder_path,
                split: str = None,
                 x_tfms: list = None,
                 y_tfms: list = None):
        
        f'''
        annotations_path: path of the json file where spatialsenses annotations are. 
        encoder_path = path of word2vec encoder
        split: 'train', 'valid' or 'test',
        x_tfms: transforms that need to be applied on the subject and object words. NOTE: no need to include transforms to convert them into vectors. 
        '''


        super().__init__()
        self.subjects = [] #x1
        self.objects = [] #x2
        self.predicates = [] #y
        
        
        
        self.split = split
        if self.split is not None: assert split in ['train', 'valid', 'test'], f"invalid selection of split. expected values = 'train', 'valid', 'test'"
        
        self.classes = list(set(spatialsenses_to_stupd.values()))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        


        # #transforms. 
        # self.embedding = phrase2vec(encoder_path, 2,300) #word2vec
        # self.phrase_encoder = PhraseEncoder(300) #to convert vectors of 2 words into one single vector. 
        assert Path(encoder_path).exists(), f'Invalid path for word2vec encoder'

        self.x_tfms = list(x_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_tfms = list(y_tfms or [noop]) + [lambda y: self.class2idx[y]]
        


        assert Path(annotations_path).exists()
        for relations in json.load(open(annotations_path)):
            if self.split and not relations["split"] == split: continue
            for relation in relations['annotations']:
                if not relation['label']: continue
                self.subjects.append(relation['subject']['name'])
                self.objects.append(relation['object']['name'])
                self.predicates.append(relation['predicate'])
        
        self.model = 'languageOnly'
    
    def __len__(self): return len(self.subjects)
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


    