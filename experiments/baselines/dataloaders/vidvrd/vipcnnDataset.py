import torch
from .utils import vidvrd_to_stupd, read_img
from .baseData import baseData
import json
import torchvision.transforms as transforms
from torchvision.io import read_video

import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import math


class vipcnnDataset(baseData):
    def __init__(self,
                annotations_directory_path,
                video_path,
                n_frames:int  = 2,
                x_tfms: list = None, 
                y_category_tfms: list = None):

        f'''

        annotations_directory_path: path of directory where vidvrd annotation json files are 
        video_path: path of directory with vidvrd mp4 images are
        n_frames = number of frames to select per relation.

        x_tfms: transforms that will be applied to the image
        y_tfms: transforms that will be applied to the predicate word
        '''
        
        super().__init__()
        assert Path(annotations_directory_path).exists(), f'invalid annotations directory path'
        assert Path(video_path).exists(), f'invalid images directory path'
        self.n_frames = n_frames


        self.subj_bbox = []
        self.obj_bbox = []
        
        self.predicates = [] #y: predicate (preposition) class name
        
        self.vid_fnames = []
        self.vid_info = [] #contains dictionaries with frames and fps of video. 
        
        
        sorted(list(set(vidvrd_to_stupd.values())))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        self.x_tfms = list(x_tfms or [noop]) + [transforms.ToTensor()]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]


        files = [f for f in Path(annotations_directory_path).iterdir() if str(f).endswith('json')]

 
        for annotations_path in files:
            annotations = json.load(open(annotations_path))
            id2obj = self._obj_id2obj(annotations)

            for relation in annotations["relation_instances"]:
                predicate = self._get_valid_predicate(relation["predicate"], vidvrd_to_stupd)
                if predicate is None: continue
                self.predicates.append(predicate)
                # self.subjects.append(id2obj[relation["subject_tid"]])
                # self.objects.append(id2obj[relation["object_tid"]])


                start_frame, end_frame = relation["begin_fid"], relation["end_fid"]
                frames = [min(f,len(annotations["trajectories"])-1) for f in  self._sample_frames(start_frame, end_frame, self.n_frames)]


                self.vid_fnames.append(Path(video_path)/f'{annotations["video_id"]}.mp4')
                self.vid_info.append({'fps': annotations["fps"], 'frames': frames})

                subj_id, obj_id = relation["subject_tid"], relation["object_tid"]
                
                bbox_s, bbox_o = [],[]
                for frame in frames: 

                    trajectory = annotations["trajectories"][frame]
                    
                    subj_bbox, obj_bbox = [0]*4, [0]*4

                    for t in trajectory: 
                        # if t["tid"]==subj_id: subj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]
                        # if t["tid"]==obj_id: obj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]
                    

                        if t["tid"]==subj_id: subj_bbox = [ t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"] ]
                        if t["tid"]==obj_id: obj_bbox = [ t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"] ]


                    bbox_s.append(subj_bbox)
                    bbox_o.append(obj_bbox)


                self.subj_bbox.append(bbox_s)
                self.obj_bbox.append(bbox_o)


        self.model = 'vipcnn'

        #miscellaneous
        self.tensor2img_tfm = transforms.ToPILImage()

    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):

        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])

        video_pth = self.vid_fnames[i]
        fps, frames = self.vid_info[i]["fps"], self.vid_info[i]["frames"]
        video_frames, _, _ = read_video(str(video_pth),pts_unit = 'sec', output_format = 'TCHW')

        
        images = []
        subj_bbox = []
        obj_bbox = []

        for k,frame in enumerate(frames):
            img = self.tensor2img_tfm(video_frames[frame])
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

    
    def __name__(self): return f'VIPCNN Model'

    
    
    
