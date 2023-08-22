import torch
from .utils import vidvrd_to_stupd, read_img, word2vec
from .baseData import baseData
import json
import torchvision.transforms as transforms
from torchvision.io import read_video


import cv2
from pathlib import Path
from PIL import Image
import numpy as np

def noop(x): return x



class vtranseDataset(baseData):
    # def __init__(self,
    #             annotations_path, 
    #             image_path ,
    #             encoder_path, 
    #             split=None, 
    #             x_category_tfms: list = None, 
    #             x_tfms: list = None,
    #             y_category_tfms: list = None):

    #     f'''
    #     annotations_path: path of spatialsense annotation json file
    #     image_path: path of directory with spatialsense images in them
    #     encoder_path: path of word2vec encoder module. 

    #     x_category_tfms: transforms that will be applied to the subject and object words (in order to get embeddings. )
    #     x_tfms: transforms that will be applied to the image
    #     y_category_tfms: transforms that will be applied to the predicate word
    #     '''
        
    #     super().__init__()
    #     assert Path(annotations_path).exists(), f'invalid annotations file path'
    #     assert Path(image_path).exists(), f'invalid images directory path'
    #     assert Path(encoder_path).exists(), f'invalid word2vec encoder path'


    #     self.subjects = [] #x1: subject classname
    #     self.objects = [] #x2: object class name
    #     self.predicates = [] #y: predicate (preposition) class name
        
    #     self.subj_bbox = []
    #     self.obj_bbox = []
        
    #     self.image_fnames = []
        
        
    #     self.split = split
    #     if self.split is not None: assert split in ['train', 'valid', 'test'], f"invalid selection of split. expected values = 'train', 'valid', 'test'"
        

    #     self.classes = list(set(spatialsenses_to_stupd.values()))
    #     self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
    #     self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
    #     self.c = len(self.classes)
        
    #     #transforms
    #     self.x_tfms = list(x_tfms or [noop]) + [transforms.ToTensor()]
    #     self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
    #     self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
 
        
    #     #enumerating all raw data objects
    #     for relations in json.load(open(annotations_path)):
    #         if self.split and not relations["split"] == split: continue
    #         for relation in relations['annotations']:
    #             if not relation['label']: continue

    #             self.predicates.append(relation['predicate'])
                
    #             self.subj_bbox.append(relation['subject']['bbox'])
    #             self.obj_bbox.append(relation['object']['bbox'])

    #             self.subjects.append(relation['subject']['name'])
    #             self.objects.append(relation['object']['name'])
                
    #             self.image_fnames.append(read_img(relations['url'], image_path))



    def __init__(self,
                annotations_directory_path,
                video_path,
                encoder_path,

                n_frames:int  = 2,
                x_category_tfms: list = None, 
                x_tfms: list = None, 
                y_category_tfms: list = None):

        f'''

        annotations_directory_path: path of directory where vidvrd annotation json files are 
        video_path: path of directory with vidvrd mp4 images are
        encoder_path: path of word2vec weight binary files.

        n_frames = number of frames to select per relation.

        x_category_tfms: transforms that will be applied to the subject and image words. 
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
        self.subjects = [] #x1: subject classname
        self.objects = [] #x2: object class name
        
        self.vid_fnames = []
        self.vid_info = [] #contains dictionaries with frames and fps of video. 
        
        
        self.classes = list(set(vidvrd_to_stupd.values()))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        self.x_category_tfms = list(x_category_tfms or [noop]) + [word2vec(encoder_path, max_phrase_len = 2, word_embedding_dim = 300)]
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
        self.x_tfms = list(x_tfms or [noop]) + [transforms.ToTensor()]


        files = [f for f in Path(annotations_directory_path).iterdir() if str(f).endswith('json')]

        for annotations_path in files:
            annotations = json.load(open(annotations_path))
            id2obj = self._obj_id2obj(annotations)

            for relation in annotations["relation_instances"]:
                predicate = self._get_valid_predicate(relation["predicate"], vidvrd_to_stupd)
                if predicate is None: continue
                self.predicates.append(predicate)
                self.subjects.append(id2obj[relation["subject_tid"]])
                self.objects.append(id2obj[relation["object_tid"]])



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

                
        self.model = 'vtranse'
        #miscellaneous
        self.tensor2img_tfm = transforms.ToPILImage()

    def __len__(self): return len(self.predicates)

    def __getitem__(self, i):
        #for language part of the model
        subj = self.apply_tfms(self.subjects[i], self.x_category_tfms)
        obj =  self.apply_tfms(self.objects[i] , self.x_category_tfms)
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_category_tfms)])
        
        # #for computer vision part of the model
        # img = Image.open(self.image_fnames[i])
        # iw,ih = img.size
            
        # image = self.apply_tfms(self._getAppr(img, [0, ih, 0, iw]), self.x_tfms)
        
        # subj_bbox  = torch.Tensor(self._fix_bbox(self.subj_bbox[i], ih, iw))
        # obj_bbox =  torch.Tensor(self._fix_bbox(self.obj_bbox[i], ih, iw))


        # t_s = torch.Tensor(self._getT(subj_bbox, obj_bbox))
        # t_o = torch.Tensor(self._getT(obj_bbox,  subj_bbox))

        video_pth = self.vid_fnames[i]
        fps, frames = self.vid_info[i]["fps"], self.vid_info[i]["frames"]
        video_frames, _, _ = read_video(str(video_pth),pts_unit = 'sec', output_format = 'TCHW')

        # bbox_img = []
        # bbox_mask = []

        images = []
        subj_bbox = []
        obj_bbox = []
        t_s, t_o = [], []

        for k,frame in enumerate(frames):
            # img = self.tensor2img_tfm(video_frames[frame])
            # ih, iw = img.shape

            img = self.tensor2img_tfm(video_frames[frame])
            ih, iw = img.shape

            image = self.apply_tfms(self._getAppr(img, [0, ih, 0, iw]), self.x_tfms)
            images.append(image)

            subj_bbox_k= torch.Tensor(self._fix_bbox(self.subj_bbox[i][k], ih, iw))
            obj_bbox_k = torch.Tensor(self._fix_bbox(self.obj_bbox[i][k],  ih, iw))

            
            subj_bbox.append(subj_bbox_k)
            obj_bbox.append(obj_bbox_k)
            t_s.append(torch.Tensor(self._getT(subj_bbox_k, obj_bbox_k)))
            t_o.append(torch.Tensor(self._getT(obj_bbox_k, subj_bbox_k)))

            # union_bbox = self._enlarge(self._getUnionBBox(subj_bbox, obj_bbox, ih, iw), 1.25, ih, iw)
            # bbox_img_k =   self.apply_tfms(self._getAppr(img, union_bbox), self.x_img_tfms)

            # bbox_mask_k = np.stack([self._getDualMask(ih, iw, subj_bbox, 32).astype(np.uint8),
            #                   self._getDualMask(ih, iw, obj_bbox,  32).astype(np.uint8),
            #                   np.zeros((32, 32), dtype=np.uint8)], 2)
            # bbox_mask_k = self.apply_tfms(bbox_mask_k, self.bbox_mask_tfms)[:2].float() / 255.0

            # bbox_img.append(bbox_img_k)
            # bbox_mask.append(bbox_mask_k)

        image, subj_bbox, obj_bbox = torch.stack(images), torch.stack(subj_bbox), torch.stack(obj_bbox)
        t_s, t_o = torch.stack(t_s), torch.stack(t_o)


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

       