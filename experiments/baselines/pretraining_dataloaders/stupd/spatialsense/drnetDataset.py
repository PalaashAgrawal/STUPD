import torch
from .utils import read_img, word2vec, stupd_classes, convert_stupdBbox_to_spatialSenseBbox
from .baseData import baseData
import json
import torchvision.transforms as transforms
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

def noop(x): return x

class drnetDataset(baseData):
    def __init__(self,
                annotations_path, 
                image_path ,
                encoder_path, 
                x_category_tfms: list = None,
                y_category_tfms: list = None,
                x_img_tfms: list = None,
                bbox_mask_tfms = None):

        f'''

        annotations_path: path of stupd annotation json file
        image_path: path of directory with stupd images in them
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
        
        
        self.classes = sorted(stupd_classes)
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        
        self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
        self.x_img_tfms = list(x_img_tfms or [noop]) + [transforms.ToTensor()]
        self.bbox_mask_tfms = list(bbox_mask_tfms or [noop]) + [transforms.ToTensor()]


        assert Path(annotations_path).exists()
        annotation_files = [o for o in annotations_path.iterdir() if str(o).endswith('csv') and o.stem in self.classes]

        for annotation in annotation_files:
            relations = pd.read_csv(annotation).dropna() #any row with incomplete data is dropped

            for k,row in relations.iterrows():

                self.predicates.append(row['relation'])
                self.subjects.append(f"{row['subject_category']} {row['subject_supercategory']}")
                self.objects.append(f"{row['object_category']} {row['object_supercategory']}")

                subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[0], eval(row['object_bbox2d'])[0]
                self.subj_bbox.append(list(convert_stupdBbox_to_spatialSenseBbox(subj_bbox)))
                self.obj_bbox.append(list(convert_stupdBbox_to_spatialSenseBbox(obj_bbox)))

                self.image_fnames.append(Path(image_path)/f"{eval(row['image_path'])[0]}")

                
        #misc
        self.img2tsr = transforms.ToTensor()
        self.tsr2img = transforms.ToPILImage()
    
    def __name__(self): return 'DRNet Model'

    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):
        #for language part of the model
        subj = torch.Tensor(self.apply_tfms(self.subjects[i], self.x_category_tfms))
        obj =  torch.Tensor(self.apply_tfms(self.objects[i] , self.x_category_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])

        
        #for computer vision part of the model
        img = Image.open(self.image_fnames[i])
        img = self.tsr2img(self.img2tsr(img)[:3])#unity saves images as RGBA images. We convert it to RGB
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
