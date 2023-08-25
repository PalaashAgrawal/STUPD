# #____________________________________________WANDB info__________________________________________
# #edit these
# project = 'SpatialSense from scratch'
# name = 'pprfcn-scratch'
# model_name = "pprfcn"
# dataset_name = 'SpatialSense'
# #___________________________________________GPU info_____________________________________________

# #Note that the optional GPU id argument given to python from command line, will override this variable

# #edit
# device_id = 0 #cuda ID for the GPU, if it exists. If you dont have GPU, ignore this. 
# #___________________________________________path info______________________________________________
# from pathlib import Path

# #edit. Expected structure: core_path contains dataset path as well as the code for the experiments. 
# core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists() 
# spatialsenses_pth = core_pth/Path('real_world_data/spatialsense'); assert spatialsenses_pth.exists()
# encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()
# stupd_path = Path('/data/dataset/agrawalp2/stupd/stupd_dataset'); assert stupd_path.exists()

# #___________________________________________run _________________________________________________
# import torch
# from fastai.distributed import *
# from fastai.vision.all import *
# import wandb
# from fastai.callback.wandb import *
# import os
# import sys

# module_path = core_pth/'experiments/baselines'; assert module_path.exists()
# if module_path not in sys.path: sys.path.append(str(module_path))

# device_id = device_id if len(sys.argv)<2 else sys.argv[1]
# os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

# device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{device_id}')

# wandb.init(project = project , name = name)




# from dataloaders.spatialsense.pprfcnDataset import pprfcnDataset
# from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd
# from models.static.pprfcn import PPRFCN
# import torchvision.transforms as transforms

# #___________________________________________code _________________________________________________

# train_ds = pprfcnDataset(annotations_path = spatialsenses_pth/'annotations.json',
#                          image_path = spatialsenses_pth/'images',
#                          split='train',
#                          x_tfms = [transforms.ToPILImage("RGB"),
#                                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
#                                   ],
#                          y_category_tfms = [map_spatialsenses_to_stupd],
#                         )

# valid_ds = pprfcnDataset(annotations_path = spatialsenses_pth/'annotations.json',
#                          image_path = spatialsenses_pth/'images',
#                          split='valid',
#                          x_tfms = [transforms.ToPILImage("RGB")],
#                          y_category_tfms = [map_spatialsenses_to_stupd])

# train_dl = DataLoader(train_ds, batch_size = 32 , shuffle = True, num_workers = 0)
# valid_dl = DataLoader(valid_ds, batch_size = 64 , shuffle = True, num_workers = 0)


# dls = DataLoaders(train_dl, valid_dl)
# dls.n_inp = 3

# model = PPRFCN(train_ds.c, imagenet_pretrained = False).cuda()
# learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
#                 path = core_pth/'experiments/baselines/weights',
#                 model_dir = model_name,
#                 cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

# learn.fit(5)



#____________________________________________WANDB info__________________________________________
#edit these
project = 'SpatialSense_scratch_STUPDpretrained'
name = 'pprfcn-scratch-STUPDpretrained'
model_name = "pprfcn"
dataset_name = 'SpatialSense'

#_____________GPU info_____________________________________________

#Note that the optional GPU id argument given to python from command line, will override this variable

#edit
device_id = 0 #cuda ID for the GPU, if it exists. If you dont have GPU, ignore this. 
#___________________________________________path info______________________________________________
from pathlib import Path

#edit. Expected structure: core_path contains dataset path as well as the code for the experiments. 
core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists() 
spatialsenses_pth = core_pth/Path('real_world_data/spatialsense'); assert spatialsenses_pth.exists()
encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()
stupd_path = Path('/data/dataset/agrawalp2/stupd/stupd_dataset'); assert stupd_path.exists()

#___________________________________________code part 1 pretraining _________________________________________________
import torch
from fastai.distributed import *
from fastai.vision.all import *
import wandb
from fastai.callback.wandb import *

import os
import sys


device_id = device_id if len(sys.argv)<2 else sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{device_id}')

wandb.init(project = project , name = name)

module_path = core_pth/'experiments/baselines'; assert module_path.exists()
if module_path not in sys.path: sys.path.append(str(module_path))





from pretraining_dataloaders.stupd.spatialsense.pprfcnDataset import pprfcnDataset
from pretraining_dataloaders.stupd.spatialsense.utils import split_dataset

from models.static.vipcnn import VipCNN
import torchvision.transforms as transforms


ds = pprfcnDataset(annotations_path = stupd_path/'annotations', 
                   image_path = stupd_path/'stupd',
                    x_tfms =      [transforms.ToPILImage("RGB"),
                                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                                  ],
                        )

train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =32 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 64 , shuffle = True, drop_last = True)

dls = DataLoaders(train_dl, valid_dl)
model = PPRFCN(train_ds.c, imagenet_pretrained = False).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
               path = core_pth/'experiments/baselines/weights',
               model_dir = model_name,
               )

learn.fit_one_cycle(2)
learn.save(name)

#___________________________________________part 2 finetuning _________________________________________________

from dataloaders.spatialsense.pprfcnDataset import pprfcnDataset
from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd

train_ds = pprfcnDataset(annotations_path = spatialsenses_pth/'annotations.json',
                         image_path = spatialsenses_pth/'images',
                         split='train',
                         x_tfms = [transforms.ToPILImage("RGB"),
                                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                                  ],
                         y_category_tfms = [map_spatialsenses_to_stupd],
                        )

valid_ds = pprfcnDataset(annotations_path = spatialsenses_pth/'annotations.json',
                         image_path = spatialsenses_pth/'images',
                         split='valid',
                         x_tfms = [transforms.ToPILImage("RGB")],
                         y_category_tfms = [map_spatialsenses_to_stupd])

train_dl = DataLoader(train_ds, batch_size =32 , shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = 64 , shuffle = True)

dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 3

model = PPRFCN(train_ds.c, imagenet_pretrained = False).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,  
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.fit(5)