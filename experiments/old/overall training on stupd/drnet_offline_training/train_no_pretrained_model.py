import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print('importing libraries')

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from fastai.distributed import *
from fastai.vision.all import *

from pathlib import Path
from functools import partial
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torchvision.models as models

from PIL import Image
import cv2

import json
from collections import OrderedDict

import ast

device = torch.device('cuda:0')
core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists()
spatialSenses_pth = core_pth/Path('real_world_data/spatialsense'); assert spatialSenses_pth.exists()
stupd_pth = Path('/mnt/dataset/agrawalp2/stupd_dataset'); assert stupd_pth.exists()

#utils
from gensim.models import KeyedVectors
import numpy as np
from autocorrect import Speller

encoder_path = spatialSenses_pth/Path('baselines/GoogleNews-vectors-negative300.bin.gz')
assert encoder_path.exists()
print('initializing word2vec encoder')
word2vec = KeyedVectors.load_word2vec_format(encoder_path, binary=True, unicode_errors="ignore")

spell = Speller()
def phrase2vec(phrase, max_phrase_len, word_embedding_dim):
    vec = np.zeros((max_phrase_len, word_embedding_dim,), dtype=np.float32)
    for i, word in enumerate(phrase.split()[:max_phrase_len]):
        if word in word2vec: vec[i] = word2vec[word]
        elif spell(word) in word2vec: vec[i] = word2vec[spell(word)]
        else: pass
    return vec


def read_img(url, imagepath):
    if url.startswith("http"):  # flickr
        filename = os.path.join(imagepath, "flickr", url.split("/")[-1])
    else:  # nyu
        filename = os.path.join(imagepath, "nyu", url.split("/")[-1])
    return filename



def noop(x): return x

def convert_stupd_bbox_to_spatialsense_bbox(bbox):
    #stupd bbox format = wmin, hmin, w, h
    #spatialsenses bbox format = hmin, hmax, wmin, wmax
    
    wmin, hmin, w,h = bbox
    return (hmin, hmin+h, wmin, wmin+w)




def is_static(x):
    static_preps = ['above', 'against_leaning', 'all_over', 'along_position', 'among', 'around_static', 
                   'behind', 'below', 'beside', 'between', 'in_front_of', 'inside', 'on', 'outside']
    
    dynamic_preps = ['against', 'along', 'around', 'by', 'down', 'from', 'into', 'into_crash', 'off', 'onto', 'out_of', 'over', 'through', 'towards', 'up', 'with']
    
    if x in static_preps: return True
    elif x in dynamic_preps: return False
    else: raise Exception("invalid relation input")





class stupd(Dataset):
    def __init__(self, 
                 annotations_path = stupd_pth/'annotations', 
                 image_path = stupd_pth/'stupd',
                 x_category_tfms: list = None,
                 y_category_tfms: list = None,
                 x_img_tfms: list = None,
                bbox_mask_tfms = None,
                frames_per_vid=10,
                ):
        
        f'''
        frames_per_vid: how many frames to consider per video for a data sample x. 
                        Per video, frames_per_vid images will be returned, using a sampling. Eg, if sampling = 'uniform', 10 uniformly spaced images will be returned. 
        
        '''
        
        super().__init__()
        self.frames_per_vid = frames_per_vid
        
        assert Path(annotations_path).exists(), f'invalid annotations file path'
        assert Path(image_path).exists(), f'invalid images directory path'
        
        self.subjects = [] #x1: subject classname
        self.objects = [] #x2: object class name
        
        self.subj_bbox = []
        self.obj_bbox = []
        
        self.predicates = [] #y: predicate (preposition) class name
        self.image_fnames = []
        self.classes = set()

        
        #enumerating all raw data objects
        for annotations in annotations_path.iterdir():
            if annotations.suffix!=".csv": continue
#             if annotations.stem not in spatialsenses_to_stupd.values(): continue
            df = self._process_df(pd.read_csv(annotations))
            for i,row in df.iterrows():
     
                subj_3dcoords = eval(row['subject_position3d'])
                obj_3dcoords = eval(row['object_position3d'])
                subj_2dbbox = eval(row['subject_bbox2d'])
                obj_2dbbox = eval(row['object_bbox2d'])
                
                img_pths = eval(row['image_path'])
                
                
                #the below if else conditions are to process the occasional missing values in the coordinate/bbox dictionaries. 
                #And also to bring static and dynamic inputs to same shape. Static images are repeated self.frames_per_vid times to emulate as if it were a static video (instead of an image)
                #if it is dynamic, we first None fill values with empty strings, then sample self.frames_per_vid images from the total 30 frames. 
                    
                if is_static(annotations.stem):
                    if subj_3dcoords is not None and len(subj_3dcoords):
                        s3d, s2dbbox = subj_3dcoords[0], subj_2dbbox[0] 
                    else: 
                        s3d, s2dbbox =  {"x":0, "y":0, "z":0},[0.,0.,0.,0.]
                    
                    if obj_3dcoords is not None and len(obj_3dcoords):
                        o3d, o2dbbox = obj_3dcoords[0], obj_2dbbox[0]  
                    else: 
                        o3d, o2dbbox = {"x":0, "y":0, "z":0},[0.,0.,0.,0.]

                        
                    self.image_fnames.append([image_path/img_pths[0]]*self.frames_per_vid)
                    self.subj_bbox.append([convert_stupd_bbox_to_spatialsense_bbox(s2dbbox) for _ in range(self.frames_per_vid)] )
                    self.obj_bbox.append([convert_stupd_bbox_to_spatialsense_bbox(o2dbbox)  for _ in range(self.frames_per_vid)])
                    
                    
                else:
                    
                    for i in range(len(subj_3dcoords)):
                        if subj_3dcoords[i] is None: subj_3dcoords[i], subj_2dbbox[i] = {"x":0, "y":0, "z":0},[0.,0.,0.,0.]
                        if obj_3dcoords[i] is None:  obj_3dcoords[i],  obj_2dbbox[i] = {"x":0, "y":0, "z":0},[0.,0.,0.,0.]
                   

                    
                    idxs = self._sample(len(img_pths))
                    self.image_fnames.append([image_path/img_pths[i] for i in idxs])
                    self.subj_bbox.append([convert_stupd_bbox_to_spatialsense_bbox(subj_2dbbox[i]) for i in idxs])
                    self.obj_bbox.append([convert_stupd_bbox_to_spatialsense_bbox(obj_2dbbox[i]) for i in idxs])
                    
                        
                self.subjects.append(row['subject_category'] + ' ' + row['subject_supercategory'])
                self.objects.append(row['object_category'] + ' ' + row['object_supercategory'])
                self.predicates.append(annotations.stem)
                self.classes.add(annotations.stem)                
                
                
        self.class2idx = {cat:i for i,cat in enumerate(self.classes)}
        self.idx2class = {self.class2idx[cat]:cat for cat in self.class2idx}
        self.c = len(self.classes)
        
        #transforms
        self.x_category_tfms = list(x_category_tfms or [noop]) 
        self.y_category_tfms = list(y_category_tfms or [noop]) + [lambda y: self.class2idx[y]]
        self.x_img_tfms = list(x_img_tfms or [noop]) + [transforms.ToTensor()]
        self.bbox_mask_tfms = list(bbox_mask_tfms or [noop]) + [transforms.ToTensor()]
        
    
    def __len__(self): return len(self.predicates)
    def __getitem__(self, i):
        #for language part of the model
        subj = self.apply_tfms(self.subjects[i], self.x_category_tfms) 
        obj =  self.apply_tfms( self.objects[i], self.x_category_tfms)
        predicate = self.apply_tfms(self.predicates[i], self.y_category_tfms)
        
        #for computer vision part of the model
        
        bbox_img, bbox_mask = [],[]
        for img,subj_bbox,obj_bbox in zip(self.image_fnames[i], self.subj_bbox[i], self.obj_bbox[i]):
            im = Image.open(img)
            ih, iw = im.shape
            union_bbox = self.enlarge(self._getUnionBBox(subj_bbox,obj_bbox, ih, iw), 1.25, ih, iw)
            bb_img = self.apply_tfms(self._getAppr(np.array(im)[:,:,:3], union_bbox), self.x_img_tfms)
            bb_mask = np.stack([self._getDualMask(ih, iw, subj_bbox , 32).astype(np.uint8),
                              self._getDualMask(ih, iw, obj_bbox , 32).astype(np.uint8),
                              np.zeros((32, 32), dtype=np.uint8)],
                             2)
            bb_mask = self.apply_tfms(bb_mask, self.bbox_mask_tfms)[:2].float() / 255.0
            
            bbox_img.append(bb_img)
            bbox_mask.append(bb_mask)
            
        
        return (torch.Tensor(subj).type(torch.cuda.FloatTensor), 
                torch.Tensor(obj).type(torch.cuda.FloatTensor), 
                torch.cat(bbox_img).type(torch.cuda.FloatTensor),
                torch.cat(bbox_mask).type(torch.cuda.FloatTensor),
                torch.Tensor([predicate]).type(torch.cuda.LongTensor))

    def apply_tfms(self, o, tfms):
        for tfm in tfms: o = tfm(o)
        return o
    
    def enlarge(self, bbox, factor, ih, iw):
        height = bbox[1] - bbox[0]
        width = bbox[3] - bbox[2]
        assert height > 0 and width > 0
        return [
            max(0, int(bbox[0] - (factor - 1.0) * height / 2.0)),
            min(ih, int(bbox[1] + (factor - 1.0) * height / 2.0)),
            max(0, int(bbox[2] - (factor - 1.0) * width / 2.0)),
            min(iw, int(bbox[3] + (factor - 1.0) * width / 2.0)),
        ]

    def _getAppr(self, im, bb, out_size=224.0):
            subim = im[bb[0] : bb[1], bb[2] : bb[3], :]
            subim = cv2.resize(
                subim,
                None,
                None,
                out_size / subim.shape[1],
                out_size / subim.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )
            subim = (subim / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] #mean and std of imagenet
            return subim.astype(np.float32, copy=False)

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
    
    def _process_df(self, df):
        f"this function replace empty values in subject/object category and supercategory columns with an empty string ('') " 
        cols = ['subject_category', 'subject_supercategory', 'object_category', 'object_supercategory']
        
        df[cols] = df[cols].fillna('')
        return df
    
    def _sample(self, num_frames=30):
        assert num_frames>=self.frames_per_vid, 'not enough values to select from. Reinitialize with reduced frames_per_vid argument'
        
        idxs = [int((i*num_frames -1)/(self.frames_per_vid-1)) for i in range(self.frames_per_vid) ]
        return idxs
    


max_phrase_len = 3 #subjects/objects can be mutliple words. eg - microwave oven. max_phrase_len decides how many words are used to create embeddings
word_embedding_dim = 300 #decides the dimension of the feature each word is converted into
frames_per_vid = 3

embedding = partial(phrase2vec, max_phrase_len = max_phrase_len, word_embedding_dim = word_embedding_dim)

    
print('creating dataset')
ds = stupd(
            x_category_tfms = [embedding],
            y_category_tfms = None,
            x_img_tfms =   [transforms.ToPILImage("RGB"),
                            transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)],

            bbox_mask_tfms = [transforms.ToPILImage("RGB"),
                            transforms.Pad(4, padding_mode="edge"),
                            transforms.RandomResizedCrop(32, scale=(0.75, 0.85))],
            
            frames_per_vid = frames_per_vid,
                         )


train_split = 0.8
train_ds, valid_ds = torch.utils.data.random_split(ds, [int(len(ds)*train_split), len(ds) - int(len(ds)*train_split)])
train_ds.c, valid_ds.c = ds.c, ds.c

print('creating DataLoaders')
train_dl = DataLoader(train_ds, batch_size =512 , shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = 512 , shuffle = False)



#__________________________________________TRAINING______________________________________-

#helper models
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.layernorm1 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels // 2, kernel_size=3, padding=1
        )
        self.layernorm2 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inp):
        x = F.relu(self.layernorm1(self.conv1(inp)))
        x = F.relu(self.layernorm2(self.conv2(x)))
        x = self.conv3(x)
        return x + self.conv_skip(inp)

class Hourglass(nn.Module):
    def __init__(self, im_size, feature_dim):
        super().__init__()
        assert im_size == 1 or im_size % 2 == 0
        self.skip_resblock = ResidualBlock(feature_dim, feature_dim, im_size)
        if im_size > 1:
            self.pre_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)
            self.layernorm1 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.sub_hourglass = Hourglass(im_size // 2, feature_dim)
            self.layernorm2 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.post_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)

    def forward(self, x):
        up = self.skip_resblock(x)
        if x.size(-1) == 1:
            return up
        down = F.max_pool2d(x, 2)
        down = F.relu(self.layernorm1(self.pre_resblock(down)))
        down = F.relu(self.layernorm2(self.sub_hourglass(down)))
        down = self.post_resblock(down)
        down = F.upsample(down, scale_factor=2)
        return up + down

    
    
class PhraseEncoder(nn.Module):
    def __init__(self, word_embedding_dim, num_layers = 1, batch_first = True, bidirectional = True):
        super().__init__()
        self.encoder = nn.GRU(input_size = word_embedding_dim, 
                                     hidden_size = word_embedding_dim//2,
                                     num_layers = num_layers,
                                     batch_first = batch_first,
                                     bidirectional = bidirectional,
                                    )
    def forward(self, x): return torch.squeeze(self.encoder(x)[0][:,-1,:])
    
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.ReLU(x)
        return x
    


class DRNet(nn.Module):
    def __init__(self, phrase_encoder, feature_dim, num_classes, num_layers=3, frames_per_vid = 10, backbone='resnet18'):
        super(DRNet, self).__init__()
        
        self.frames_per_vid = frames_per_vid
        
        self.phrase_encoder =  phrase_encoder
        self.feature_dim = feature_dim
        
        self.appr_module = models.__dict__[backbone](pretrained=False)

        self.appr_module.conv1 = nn.Conv2d(3*self.frames_per_vid, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.appr_module.fc = nn.Linear(512, feature_dim//2)
        
        self.num_layers = num_layers
        self.c = num_classes
        
        

        self.pos_module = nn.Sequential(
            OrderedDict(
                [
                    ("conv1_p", nn.Conv2d(2*self.frames_per_vid, 32, 5, 2, 2)),
                    ("batchnorm1_p", nn.BatchNorm2d(32)),
                    ("relu1_p", nn.ReLU()),
                    ("conv2_p", nn.Conv2d(32, 64, 3, 1, 1)),
                    ("batchnorm2_p", nn.BatchNorm2d(64)),
                    ("relu2_p", nn.ReLU()),
                    ("maxpool2_p", nn.MaxPool2d(2)),
                    ("hg", Hourglass(8, 64)),
                    ("batchnorm_p", nn.BatchNorm2d(64)),
                    ("relu_p", nn.ReLU()),
                    ("maxpool_p", nn.MaxPool2d(2)),
                    ("conv3_p", nn.Conv2d(64, feature_dim//2, 4)),
                    ("batchnorm3_p", nn.BatchNorm2d(feature_dim//2)),
                ]
            )
        )

        self.batchnorm = nn.BatchNorm1d(feature_dim)
        self.Phi_subj = nn.Linear(300, feature_dim//2)
        self.Phi_obj = nn.Linear(300, feature_dim//2)

        self.fc = nn.Sequential(LinearBlock(2*feature_dim,feature_dim), 
                                LinearBlock(feature_dim, feature_dim//2), 
                                nn.Linear(feature_dim//2, self.c))

    def forward(self, subj, obj, im, posdata):
        appr_feature = self.appr_module(im) #output 256 features
        pos_feature = (self.pos_module(posdata).view(-1, self.feature_dim//2))#output 256 features
        
        
        qa = self.phrase_encoder(subj)
        qb = self.phrase_encoder(obj)
        
        x = torch.cat([appr_feature, 
                       pos_feature,
                       self.Phi_subj(qa), self.Phi_obj(qb),
                      ], 1)

        x = self.fc(x)
        return x


dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 4

phrase_encoder = PhraseEncoder(300)

print('initializing model and learner class')
model = DRNet(phrase_encoder, 512, train_ds.c, frames_per_vid = frames_per_vid).cuda()
learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()])


print('Starting Training')
# with learn.no_bar():
learn.fit(5, 1e-3)

print('\n\n\n\n\n\n')

print('accuracies')
print(learn.recorder.values)

learn.save(core_pth/'experiments/overall training on stupd/models/drnet_stupd_only.pkl')



corrects = [0]*ds.c
total = [0]*ds.c
accs = {}

corrects_relation_super = [0]*2
total_relation_super = [0]*2
accs_relation_super = [0]*2

for i,(a,b,c,d,e) in enumerate(valid_dl):
    print(i, end = ' ')
    res = torch.argmax(model(a,b,c,d), dim = 1, keepdim=False)
    
    for gt,r in zip(e, res): 
        total[gt]+=1
        corrects[gt]+=(int(gt==r))
        
        total_relation_super[is_static(ds.idx2class[gt.item()])]+=1
        corrects_relation_super[is_static(ds.idx2class[gt.item()])]+=(int(gt==r))
    
    
for i in range(len(corrects)):
    accs[ds.idx2class[i]]=corrects[i]/total[i]
for i in range(len(corrects_relation_super)):
    accs_relation_super[i]=corrects_relation_super[i]/total_relation_super[i]
    

fig = plt.figure(figsize = (25, 5))
# creating the bar plot
plt.bar(accs.keys(), accs.values(), width = 0.4, color='pink')
 
plt.xlabel("relation")
plt.xticks(rotation = 90)
plt.ylabel("accuracy")
plt.title("DRNET accuracy")

plt.tick_params(axis="y",direction="in", pad=-22)
plt.tick_params(axis="x",direction="in", pad=-100)
plt.savefig(core_pth/'experiments/overall training on stupd/drnet_offline_training/class_wise_accuracy.png')


print('class wise accuracies')
print(accs)


fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(['static', 'dynamic'], accs_relation_super, color = 'pink', width = 0.4)

plt.xlabel("relation")
plt.xticks(rotation = 90)
plt.ylabel("accuracy")
plt.title("DRNET aacuracy")
plt.tick_params(axis="y",direction="in", pad=-22)
plt.tick_params(axis="x",direction="in", pad=-100)

plt.savefig(core_pth/'experiments/overall training on stupd/drnet_offline_training/static_vs_dynamic_accuracy.png')


print('static vs dynamic accuracy')
print('static_accuracy', accs_relation_super[0])
print('dynamic_accuracy', accs_relation_super[1])

print('sanity check - does weighted mean of these give overall accuracy?')
print(accs_relation_super[1]*16/30 + accs_relation_super[0]*14/30)
