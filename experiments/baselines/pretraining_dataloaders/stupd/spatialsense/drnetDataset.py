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

class drnetDataset(baseData):
    def __init__(self,
                annotations_path, 
                image_path ,
                encoder_path, 
                split=None, 
                x_category_tfms: list = None,
                y_category_tfms: list = None,
                x_img_tfms: list = None,
                bbox_mask_tfms = None):

        f'''

        annotations_path: path of spatialsense annotation json file
        image_path: path of directory with spatialsense images in them
        encoder_path: path of word2vec encoder module. 

        x_category_tfms: transforms that will be applied to the subject/object words
        y_category_tfms: transforms that will be applied to the predicate word
        x_img_tfms: transforms that will be applied to the image
        bbox_mask_tfms: transforms that will be applied to subject and object boundinb box images. 
        '''
        
        super().__init__()
        assert Path(annotations_path).exists(), f'invalid annotations file path'
        assert Path(image_path).exists(), f'invalid images directory path'
        assert Path(encoder_path).exists(), f'invalid word2vec encoder path'
        
        self.subjects = [] #x1: subject classname
        self.objects = [] #x2: object class name
        
        self.subj_bbox = []
        self.obj_bbox = []
        
        self.predicates = [] #y: predicate (preposition) class name
        
        self.image_fnames = []
        
        
        self.split = split
        if self.split is not None: assert split in ['train', 'valid', 'test'], f"invalid selection of split. expected values = 'train', 'valid', 'test'"
        
        self.classes = list(set(spatialsenses_to_stupd.values()))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        

        self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
        self.x_img_tfms = list(x_img_tfms or [noop]) + [transforms.ToTensor()]
        self.bbox_mask_tfms = list(bbox_mask_tfms or [noop]) + [transforms.ToTensor()]
        
        #enumerating all raw data objects
        for relations in json.load(open(annotations_path)):
            if self.split and not relations["split"] == split: continue
            for relation in relations['annotations']:
                if not relation['label']: continue
                self.subjects.append(relation['subject']['name'])
                self.objects.append(relation['object']['name'])
                self.predicates.append(relation['predicate'])
                
                self.subj_bbox.append(relation['subject']['bbox'])
                self.obj_bbox.append(relation['object']['bbox'])
                
                self.image_fnames.append(read_img(relations['url'], image_path))
                
    
    def __name__(self): return 'DRNet Model'

    def __len__(self): return len(self.subjects)

    def __getitem__(self, i):
        #for language part of the model
        subj = torch.Tensor(self.apply_tfms(self.subjects[i], self.x_category_tfms))
        obj =  torch.Tensor(self.apply_tfms(self.objects[i] , self.x_category_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])

        
        #for computer vision part of the model
        img = Image.open(self.image_fnames[i])
        ih, iw = img.shape 

        #PAg: Honestly, self.fix_bbox is a pointless and unnecessary engineering function. 
        subj_bbox= self._fix_bbox(self.subj_bbox[i], ih, iw)
        obj_bbox = self._fix_bbox(self.obj_bbox[i],  ih, iw)

        union_bbox = self._enlarge(self._getUnionBBox(subj_bbox, obj_bbox, ih, iw), 1.25, ih, iw)
        bbox_img =   self.apply_tfms(self._getAppr(img, union_bbox), self.x_img_tfms)
        

        bbox_mask = np.stack([self._getDualMask(ih, iw, subj_bbox, 32).astype(np.uint8),
                              self._getDualMask(ih, iw, obj_bbox,  32).astype(np.uint8),
                              np.zeros((32, 32), dtype=np.uint8)], 2)
        bbox_mask = self.apply_tfms(bbox_mask, self.bbox_mask_tfms)[:2].float() / 255.0
    
        if torch.cuda.is_available():
            subj,obj, bbox_img, bbox_mask, predicate = (subj.type(torch.cuda.FloatTensor), 
                                                        obj.type(torch.cuda.FloatTensor), 
                                                        bbox_img.type(torch.cuda.FloatTensor),
                                                        bbox_mask.type(torch.cuda.FloatTensor),
                                                        predicate.type(torch.cuda.LongTensor))

        return subj,obj, bbox_img, bbox_mask, predicate

   