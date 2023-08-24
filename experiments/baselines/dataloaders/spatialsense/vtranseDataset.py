import torch
from .utils import spatialsenses_to_stupd, read_img, word2vec
from .baseData import baseData
import json
import torchvision.transforms as transforms
import cv2
from pathlib import Path
from PIL import Image
import numpy as np

def noop(x): return x



class vtranseDataset(baseData):
    def __init__(self,
                annotations_path, 
                image_path ,
                encoder_path, 
                split=None, 
                x_category_tfms: list = None, 
                x_tfms: list = None,
                y_category_tfms: list = None):

        f'''
        annotations_path: path of spatialsense annotation json file
        image_path: path of directory with spatialsense images in them
        encoder_path: path of word2vec encoder module. 

        x_category_tfms: transforms that will be applied to the subject and object words (in order to get embeddings. )
        x_tfms: transforms that will be applied to the image
        y_category_tfms: transforms that will be applied to the predicate word
        '''
        
        super().__init__()
        assert Path(annotations_path).exists(), f'invalid annotations file path'
        assert Path(image_path).exists(), f'invalid images directory path'
        assert Path(encoder_path).exists(), f'invalid word2vec encoder path'


        self.subjects = [] #x1: subject classname
        self.objects = [] #x2: object class name
        self.predicates = [] #y: predicate (preposition) class name
        
        self.subj_bbox = []
        self.obj_bbox = []
        
        self.image_fnames = []
        
        
        self.split = split
        if self.split is not None: assert split in ['train', 'valid', 'test'], f"invalid selection of split. expected values = 'train', 'valid', 'test'"
        

        self.classes = sorted(list(set(spatialsenses_to_stupd.values())))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        self.x_tfms = list(x_tfms or [noop]) + [transforms.ToTensor()]
        self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
 
        
        #enumerating all raw data objects
        for relations in json.load(open(annotations_path)):
            if self.split and not relations["split"] == split: continue
            for relation in relations['annotations']:
                if not relation['label']: continue

                self.predicates.append(relation['predicate'])
                
                self.subj_bbox.append(relation['subject']['bbox'])
                self.obj_bbox.append(relation['object']['bbox'])

                self.subjects.append(relation['subject']['name'])
                self.objects.append(relation['object']['name'])
                
                self.image_fnames.append(read_img(relations['url'], image_path))
                
        self.model = 'vtranse'

    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):
        #for language part of the model
        subj = self.apply_tfms(self.subjects[i], self.x_category_tfms)
        obj =  self.apply_tfms(self.objects[i] , self.x_category_tfms)
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])
        
        #for computer vision part of the model
        img = Image.open(self.image_fnames[i])
        ih, iw = img.shape
            
        image = self.apply_tfms(self._getAppr(img, [0, ih, 0, iw]), self.x_tfms)
        
        subj_bbox  = torch.Tensor(self._fix_bbox(self.subj_bbox[i], ih, iw))
        obj_bbox =  torch.Tensor(self._fix_bbox(self.obj_bbox[i], ih, iw))


        t_s = torch.Tensor(self._getT(subj_bbox, obj_bbox))
        t_o = torch.Tensor(self._getT(obj_bbox,  subj_bbox))

        if torch.cuda.is_available():
            subj, obj, image, t_s, t_o, subj_bbox, obj_bbox, predicate = (
                                                                        torch.Tensor(subj).type(torch.cuda.FloatTensor), 
                                                                        torch.Tensor(obj).type(torch.cuda.FloatTensor), 

                                                                        image.type(torch.cuda.FloatTensor), 
                                                                        
                                                                        t_s.type(torch.cuda.FloatTensor),
                                                                        t_o.type(torch.cuda.FloatTensor),

                                                                        subj_bbox.type(torch.cuda.FloatTensor), 
                                                                        obj_bbox.type(torch.cuda.FloatTensor),
                                                                        
                                                                        predicate.type(torch.cuda.LongTensor))

        return subj, obj, image, t_s, t_o, subj_bbox, obj_bbox, predicate

    
    def __name__(self): return f'VTransE Model'

       