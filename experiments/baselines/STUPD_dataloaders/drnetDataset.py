import torch
from .utils import stupd_classes, read_img, word2vec, convert_stupdBbox_to_spatialSenseBbox
from .baseData import baseData
import json
import torchvision.transforms as transforms
from torchvision.io import read_video

import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

def noop(x): return x

class drnetDataset(baseData):
    def __init__(self,
                annotations_path, 
                video_path,
                encoder_path, 
                n_frames:int  = 2,
                x_category_tfms: list = None,
                y_category_tfms: list = None,
                x_img_tfms: list = None,
                bbox_mask_tfms = None):

        f'''

        annotations_path: path of directory where vidvrd annotation json files are 
        video_path: path of directory with vidvrd mp4 images are
        encoder_path: path of word2vec encoder module. 
        n_frames = number of frames to select per relation.

        x_category_tfms: transforms that will be applied to the subject/object words
        y_category_tfms: transforms that will be applied to the predicate word
        x_img_tfms: transforms that will be applied to the image
        bbox_mask_tfms: transforms that will be applied to subject and object boundinb box images. 
        '''
        
        super().__init__()
        assert Path(annotations_path).exists(), f'invalid annotations file path'
        assert Path(video_path).exists(), f'invalid images directory path'
        assert Path(encoder_path).exists(), f'invalid word2vec encoder path'
        self.n_frames = n_frames
        
        self.subjects = [] #x1: subject classname
        self.objects = [] #x2: object class name
        
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
        
        self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
        self.x_img_tfms = list(x_img_tfms or [noop]) + [transforms.ToTensor()]
        self.bbox_mask_tfms = list(bbox_mask_tfms or [noop]) + [transforms.ToTensor()]


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


        #         start_frame, end_frame = relation["begin_fid"], relation["end_fid"]
        #         frames = [min(f,len(annotations["trajectories"])-1) for f in  self._sample_frames(start_frame, end_frame, self.n_frames)]


        #         self.vid_fnames.append(Path(video_path)/f'{annotations["video_id"]}.mp4')
        #         self.vid_info.append({'fps': annotations["fps"], 'frames': frames})

        #         subj_id, obj_id = relation["subject_tid"], relation["object_tid"]
                
        #         bbox_s, bbox_o = [],[]
        #         for frame in frames: 

        #             trajectory = annotations["trajectories"][frame]
                    
        #             subj_bbox, obj_bbox = [0]*4, [0]*4
        #             # subj_bbox, obj_bbox = None, None

        #             for t in trajectory: 
        #                 # if t["tid"]==subj_id: subj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]
        #                 # if t["tid"]==obj_id: obj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]
                    

        #                 if t["tid"]==subj_id: subj_bbox = [ t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"] ]
        #                 if t["tid"]==obj_id: obj_bbox = [ t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"] ]


        #             bbox_s.append(subj_bbox)
        #             bbox_o.append(obj_bbox)


        #         self.subj_bbox.append(bbox_s)
        #         self.obj_bbox.append(bbox_o)

        assert Path(annotations_path).exists()
        annotation_files = [o for o in annotations_path.iterdir() if str(o).endswith('csv') and o.stem in self.classes]

        for annotation in annotation_files:
            relations = pd.read_csv(annotation).dropna() #any row with incomplete data is dropped
            
            for k,row in relations.iterrows():
                fnames_list = eval(row['image_path'])
                start_frame, end_frame = 0, len(fnames_list)-1
                frames = self._sample_frames(start_frame, end_frame, self.n_frames)
                
                self.predicates.append(row['relation'])
                self.subjects.append(f"{row['subject_category']} {row['subject_supercategory']}")
                self.objects.append(f"{row['object_category']} {row['object_supercategory']}")

                # coords = []
                bbox_s, bbox_o = [],[]
                fnames = []

                for frame in frames:
                    subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[frame], eval(row['object_bbox2d'])[frame]
                    while not subj_bbox or not obj_bbox: #some are None values. 
                        frame+=1
                        subj_bbox, obj_bbox = eval(row['subject_bbox2d'])[frame%end_frame], eval(row['object_bbox2d'])[frame%end_frame]
                    # subj_position, obj_position = eval(row['subject_position3d'])[frame], eval(row['object_position3d'])[frame]

                    # coords.append([subj_position['x']-obj_position['x'],
                    #                 subj_position['y']-obj_position['y'],
                    #                 *convert_stupdBbox_to_spatialSenseBbox(subj_bbox),
                    #                 *convert_stupdBbox_to_spatialSenseBbox(obj_bbox),
                    #                 ])
                    bbox_s.append(convert_stupdBbox_to_spatialSenseBbox(subj_bbox))
                    bbox_o.append(convert_stupdBbox_to_spatialSenseBbox(obj_bbox))

                    img_path = video_path/f'{fnames_list[frame]}' ; assert img_path.exists()
                    
                    fnames.append(video_path/f'{fnames_list[frame]}')

                
                self.subj_bbox.append(bbox_s)
                self.obj_bbox.append(bbox_o)
                self.fnames.append(fnames)



                # self.coords.append(coords)



        self.model = 'drnet'

        #miscellaneous
        self.img2tsr = transforms.ToTensor()
        self.tsr2img = transforms.ToPILImage()
                
    
    def __name__(self): return 'DRNet Model'

    def __len__(self): return len(self.predicates)


    def __getitem__(self, i):
        #for language part of the model
        subj = torch.Tensor(self.apply_tfms(self.subjects[i], self.x_category_tfms))
        obj =  torch.Tensor(self.apply_tfms(self.objects[i] , self.x_category_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])




        # video_pth = self.vid_fnames[i]
        # fps, frames = self.vid_info[i]["fps"], self.vid_info[i]["frames"]
        # video_frames, _, _ = read_video(str(video_pth),pts_unit = 'sec', output_format = 'TCHW')
        frames = self.fnames[i]

        bbox_img = []
        bbox_mask = []
        
        for k,frame in enumerate(frames):
            img = Image.open(frame)
            img = self.tsr2img(self.img2tsr(img)[:3])#unity saves images as RGBA images. We convert it to RGB
            ih, iw = img.shape

            subj_bbox= self._fix_bbox(self.subj_bbox[i][k], ih, iw)
            obj_bbox = self._fix_bbox(self.obj_bbox[i][k],  ih, iw)

            union_bbox = self._enlarge(self._getUnionBBox(subj_bbox, obj_bbox, ih, iw), 1.25, ih, iw)
            bbox_img_k =   self.apply_tfms(self._getAppr(img, union_bbox), self.x_img_tfms)

            bbox_mask_k = np.stack([self._getDualMask(ih, iw, subj_bbox, 32).astype(np.uint8),
                              self._getDualMask(ih, iw, obj_bbox,  32).astype(np.uint8),
                              np.zeros((32, 32), dtype=np.uint8)], 2)
            bbox_mask_k = self.apply_tfms(bbox_mask_k, self.bbox_mask_tfms)[:2].float() / 255.0

            bbox_img.append(bbox_img_k)
            bbox_mask.append(bbox_mask_k)
        
        bbox_img = torch.stack(bbox_img)
        bbox_mask = torch.stack(bbox_mask)
        bbox_mask = bbox_mask.view(-1, bbox_mask.shape[-2], bbox_mask.shape[-1])

        if torch.cuda.is_available():
            subj,obj, bbox_img, bbox_mask, predicate = (subj.type(torch.cuda.FloatTensor), 
                                                        obj.type(torch.cuda.FloatTensor), 
                                                        bbox_img.type(torch.cuda.FloatTensor),
                                                        bbox_mask.type(torch.cuda.FloatTensor),
                                                        predicate.type(torch.cuda.LongTensor))

        return subj,obj, bbox_img, bbox_mask, predicate
