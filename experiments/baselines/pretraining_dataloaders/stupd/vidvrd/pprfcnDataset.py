import torch
from .utils import stupd_classes, read_img, convert_stupdBbox_to_spatialSenseBbox
from .baseData import baseData
import json
import torchvision.transforms as transforms
from torchvision.io import read_video

import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import math
import pandas as pd

def noop(x): return x

class pprfcnDataset(baseData):
    def __init__(self,
                annotations_path,
                video_path,
                n_frames:int  = 2,
                x_tfms: list = None, 
                y_category_tfms: list = None):

        f'''

        annotations_path: path of directory where vidvrd annotation json files are 
        video_path: path of directory with vidvrd mp4 images are
        n_frames = number of frames to select per relation.

        x_tfms: transforms that will be applied to the image
        y_tfms: transforms that will be applied to the predicate word
        '''
        
        super().__init__()
        assert Path(annotations_path).exists(), f'invalid annotations directory path'
        assert Path(video_path).exists(), f'invalid images directory path'
        self.n_frames = n_frames


        self.subj_bbox = []
        self.obj_bbox = []
        
        self.predicates = [] #y: predicate (preposition) class name
        
        # self.vid_fnames = []
        # self.vid_info = [] #contains dictionaries with frames and fps of video. 
        self.fnames = []
        
        self.classes = sorted(stupd_classes)
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        self.x_tfms = list(x_tfms or [noop]) + [transforms.ToTensor()]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]


        assert Path(annotations_path).exists()
        annotation_files = [o for o in annotations_path.iterdir() if str(o).endswith('csv') and o.stem in self.classes]

        for annotation in annotation_files:
            relations = pd.read_csv(annotation).dropna() #any row with incomplete data is dropped
            
            for k,row in relations.iterrows():
                fnames_list = eval(row['image_path'])
                start_frame, end_frame = 0, len(fnames_list)-1
                frames = self._sample_frames(start_frame, end_frame, self.n_frames)
                
                self.predicates.append(row['relation'])


                # coords = []
                bbox_s, bbox_o = [],[]
                fnames = []

                for frame in frames:
                    subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[frame], eval(row['object_bbox2d'])[frame]
                    while not subj_bbox or not obj_bbox: #some are None values. 
                        frame+=1
                        subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[frame%end_frame], eval(row['object_bbox2d'])[frame%end_frame]

                    bbox_s.append(convert_stupdBbox_to_spatialSenseBbox(subj_bbox))
                    bbox_o.append(convert_stupdBbox_to_spatialSenseBbox(obj_bbox))

                    img_path = video_path/f'{fnames_list[frame]}' ; assert img_path.exists()
                    
                    fnames.append(video_path/f'{fnames_list[frame]}')

                
                self.subj_bbox.append(bbox_s)
                self.obj_bbox.append(bbox_o)
                self.fnames.append(fnames)



                # self.coords.append(coords)


        self.model = 'pprfcn'

        #miscellaneous
        self.img2tsr = transforms.ToTensor()
        self.tsr2img = transforms.ToPILImage()
    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):

        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])

        frames = self.fnames[i]
        
        images = []
        subj_bbox = []
        obj_bbox = []

        for k,frame in enumerate(frames):
            # img = self.tensor2img_tfm(video_frames[frame])
            img = Image.open(frame)
            img = self.tsr2img(self.img2tsr(img)[:3])#unity saves images as RGBA images. We convert it to RGB
            ih, iw = img.shape
            image = self.apply_tfms(self._getAppr(img, [0, ih, 0, iw]), self.x_tfms)

            subj_bbox_k= torch.Tensor(self._fix_bbox(self.subj_bbox[i][k], ih, iw))
            obj_bbox_k = torch.Tensor(self._fix_bbox(self.obj_bbox[i][k],  ih, iw))

            images.append(image)
            subj_bbox.append(subj_bbox_k)
            obj_bbox.append(obj_bbox_k)
        
        image, subj_bbox, obj_bbox = torch.stack(images), torch.stack(subj_bbox), torch.stack(obj_bbox)


        if torch.cuda.is_available():
            image, subj_bbox, obj_bbox, predicate = (image.type(torch.cuda.FloatTensor), 
                                                    subj_bbox.type(torch.cuda.FloatTensor), 
                                                    obj_bbox.type(torch.cuda.FloatTensor),
                                                    predicate.type(torch.cuda.LongTensor))

        return image, subj_bbox, obj_bbox, predicate

    def __name__(self): return f'PPRFCN Model'

    