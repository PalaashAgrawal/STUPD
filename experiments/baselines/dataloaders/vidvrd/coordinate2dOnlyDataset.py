import torch
from torch.utils.data import Dataset
from .utils import vidvrd_to_stupd
from .baseData import baseData

import json
from pathlib import Path

def noop(x): return x


def get_center(bbox):
    hmin, hmax, wmin, wmax = bbox
    return int((wmin+wmax)/2), int((hmin+hmax)/2)

class coordinate2D_OnlyDataset(baseData):
    def __init__(self,
                annotations_directory_path, 
                n_frames: int = None,
                x_tfms: list = None,
                y_tfms: list = None):
            
        super().__init__()

        self.n_frames = n_frames

        self.coords = []
        self.predicates = [] #y

        
        
        
        self.classes = sorted(list(set(vidvrd_to_stupd.values())))
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        self.x_tfms = list(x_tfms or [noop]) 
        self.y_tfms = list(y_tfms or [noop]) + [lambda y: self.class2idx[y]]
        

        assert Path(annotations_directory_path).exists()
        files = [f for f in Path(annotations_directory_path).iterdir() if str(f).endswith('json')]

        
        start_end_frame_stamps = []


        for annotations_path in files:
            annotations = json.load(open(annotations_path))
            id2obj = self._obj_id2obj(annotations)

            for relation in annotations["relation_instances"]:
                predicate = self._get_valid_predicate(relation["predicate"], vidvrd_to_stupd)
                if predicate is None: continue
                self.predicates.append(predicate)
                
                start_frame, end_frame = relation["begin_fid"], relation["end_fid"]
                # frames = self._sample_frames(start_frame, end_frame, self.n_frames)
                frames = [min(f,len(annotations["trajectories"])-1) for f in  self._sample_frames(start_frame, end_frame, self.n_frames)]

                subj_id, obj_id = relation["subject_tid"], relation["object_tid"]

                coords = []
                # subj_bbox, obj_bbox = [0]*4, [0]*4 #initialization
                subj_bbox, obj_bbox = None, None

                for frame in frames: 
                    # frame = min(frame, len(annotations["trajectories"])-1)
                    trajectory = annotations["trajectories"][frame]
                    for t in trajectory: 
                        # if t["tid"]==subj_id: subj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]
                        # if t["tid"]==obj_id: obj_bbox = t["bbox"]["ymin"]/annotations["height"], t["bbox"]["ymax"]/annotations["height"], t["bbox"]["xmin"]/annotations["width"], t["bbox"]["xmax"]/annotations["width"]

                        if t["tid"]==subj_id: subj_bbox = t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"]
                        if t["tid"]==obj_id: obj_bbox = t["bbox"]["ymin"], t["bbox"]["ymax"], t["bbox"]["xmin"], t["bbox"]["xmax"]
                    
                    subjx, subjy = get_center(subj_bbox)
                    objx, objy = get_center(obj_bbox)
                    
                    coords.extend([subjx - objx,
                                    subjy - objy, 
                                    *subj_bbox,
                                    *obj_bbox
                                    ])



                # self.subjects.append(id2obj[relation["subject_tid"]])
                # self.objects.append(id2obj[relation["object_tid"]])

                self.coords.append(coords)




        # assert Path(annotations_path).exists()
        # for relations in json.load(open(annotations_path)):
        #     if self.split and not relations["split"] == split: continue
        #     for relation in relations['annotations']:
        #         if not relation['label']: continue

        #         self.coords.append([relation['subject']['x'] - relation['object']['x'], 
        #                             relation['subject']['y'] - relation['object']['y'], 
        #                             *relation['subject']['bbox'], 
        #                             *relation['object']['bbox']])


        #         self.predicates.append(relation['predicate'])

        self.model = 'coordinateOnly'
    
    def __len__(self): return len(self.coords)

    def __getitem__(self, i):
        coord = torch.Tensor(self.apply_tfms(self.coords[i], self.x_tfms))
        predicate = torch.Tensor([self.apply_tfms(self.predicates[i], self.y_tfms)])


    
        if torch.cuda.is_available(): coord, predicate = coord.type(torch.cuda.FloatTensor), predicate.type(torch.cuda.LongTensor)
        
        return (coord, predicate)
        

    