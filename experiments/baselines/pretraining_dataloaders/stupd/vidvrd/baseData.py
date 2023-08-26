from torch.utils.data import Dataset
import cv2
import numpy as np
import math
import PIL
from torchvision import transforms 
import torch


class baseData(Dataset):

    def __init__(self):
        super().__init__()
        self.model = ''

        self.image2tensor = transforms.PILToTensor()
        self.normalize_imagenetStats = transforms.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225])



    def _obj_id2obj(self, relation):
        f'''
        in vidvrd, json file contais a key named "subject/objects", containing object name info in the format ["category": "obj_1", "tid": 0, "category": "obj_2", "tid": 1]
        ''' 

        obj_field = relation["subject/objects"]
        ret = {}
        for o in obj_field: ret[o["tid"]] = o["category"]
        return ret

    def _get_valid_predicate(self, predicate, mapping_dict):
        f'''
        For vidvrd dataset only. 
        if predicate category is related to stupd (determined by the mapping dict), then we replace it with a valid stupd preposition name. 
        Else we return None
        '''
        res = None
        for valid_predicate in mapping_dict:
            if valid_predicate in predicate: return valid_predicate

        return res



    def _sample_frames(self, start, end, n_frames):
        f'''
        function to uniformly samples n_frames between start and end, inclusive
        '''
        
        if start==end: return [start]*n_frames
        assert n_frames < end-start+1, f'cant sample {n_frames} frames from end-start+1 = {end-start+1} frames. Please ensure that n_frames = end-start+1 OR end == start'
        if n_frames is None: n_frames = end-start+1 #select all the frames
        return [start + int(i*(end-start)/(self.n_frames -1)) for i in range(n_frames)]

     
    def _enlarge(self, bbox, factor, ih, iw):
        height = abs(bbox[1] - bbox[0]) + 1e-6
        width = abs(bbox[3] - bbox[2]) + 1e-6
        # assert height > 0 and width > 0
        return [
            max(0, int(bbox[0] - (factor - 1.0) * height / 2.0)),
            min(ih, int(bbox[1] + (factor - 1.0) * height / 2.0)),
            max(0, int(bbox[2] - (factor - 1.0) * width / 2.0)),
            min(iw, int(bbox[3] + (factor - 1.0) * width / 2.0)),
        ]

    def _getUnionBBox(self, aBB, bBB, ih, iw, margin=10):
            return [max(0, min(aBB[0], bBB[0]) - margin),
                    min(ih, max(aBB[1], bBB[1]) + margin),
                    max(0, min(aBB[2], bBB[2]) - margin),
                    min(iw, max(aBB[3], bBB[3]) + margin)]
        
    def _getDualMask(self, ih, iw, bb, heatmap_size=32):
            rh = float(heatmap_size) / ih
            rw = float(heatmap_size) / iw
            x1 = max(0, int(math.floor(bb[0] * rh)))
            x2 = min(heatmap_size, int(math.ceil(bb[1] * rh)))
            y1 = max(0, int(math.floor(bb[2] * rw)))
            y2 = min(heatmap_size, int(math.ceil(bb[3] * rw)))
            mask = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
            mask[x1:x2, y1:y2] = 255
            return mask


    def _getAppr(self, image, bbox):
            hmin, hmax, wmin, wmax = bbox

            #there are some anomalies in coordinates sometimes, where wmin>wmax, and hmin>hmax. 
            #I suspect that this must be because of wrong order of coordinates. But I'm not sure
            if hmin>hmax: hmin, hmax = hmax, hmin
            if wmin>wmax: wmin,wmax = wmax, wmin 

            out_size = self._getOutSize()
            
            sub_image = image.crop((wmin, hmin, wmax, hmax)).resize((out_size, out_size))

            sub_image = self.image2tensor(sub_image)
            sub_image = self.normalize_imagenetStats(sub_image/255.0)
            return sub_image

    
    def _getOutSize(self):
        f'''
        Different models have different image sizes as input (according to the spatialsense repository). 
        This function is a helper function for self._getAppr 
        '''
        if "vipcnn" in self.model: img_size = 400
        elif "pprfcn" in self.model: img_size = 720
        else: img_size = 224

        return img_size


    def _fix_bbox(self, bbox, ih, iw):
        f'There are some mistakes in annotations, where bbox is too small OR hmin > hmax (or wmin>wmax). This is a function that doesnt give accurate bboxes, but circumvents errors'
        if bbox[1] - bbox[0] < 20:
            if bbox[0] > 10: bbox[0] -= 10
            if bbox[1] < ih - 10: bbox[1] += 10

        if bbox[3] - bbox[2] < 20:
            if bbox[2] > 10: bbox[2] -= 10
            if bbox[3] < iw - 10: bbox[3] += 1
        return bbox

    def _getT(self, bbox1, bbox2):
                f'''
                This transform is required by VTransE model. 
                '''
                h1 = bbox1[1] - bbox1[0]
                w1 = bbox1[3] - bbox1[2]
                h2 = bbox2[1] - bbox2[0]
                w2 = bbox2[3] - bbox2[2]
                return [
                    (bbox1[0] - bbox2[0]) / float(h2),
                    (bbox1[2] - bbox2[2]) / float(w2),
                    math.log(h1 / float(h2)),
                    math.log(w1 / float(w2)),
                ]





    def apply_tfms(self, o, tfms):
        for tfm in tfms: o = tfm(o)
        return o



    
    