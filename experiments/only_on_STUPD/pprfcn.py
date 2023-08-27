#____________________________________________WANDB info__________________________________________
#edit these
project = 'directly on STUPD'
name = 'pprfcn'
model_name = "pprfcn"
dataset_name = 'ViDVRD'
#___________________________________________GPU info_____________________________________________

#Note that the optional GPU id argument given to python from command line, will override this variable

#edit
device_id = 0 #cuda ID for the GPU, if it exists. If you dont have GPU, ignore this. 
#___________________________________________path info______________________________________________
from pathlib import Path

#edit. Expected structure: core_path contains dataset path as well as the code for the experiments. 
core_pth = Path('/home/user/prepositions'); assert core_pth.exists()
vidvrd_path = core_pth/Path('real_world_data/vidvrd/vidvrd-dataset'); assert vidvrd_path.exists()
encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()
stupd_path = core_pth/Path('stupd_backup/stupd_dataset'); assert stupd_path.exists()


#___________________________________________num_frames____________________________________________
#edit
n_frames = 3


#___________________________________________run _________________________________________________
import torch
from fastai.distributed import *
from fastai.vision.all import *
import wandb
from fastai.callback.wandb import *
import os
import sys

module_path = core_pth/'experiments/baselines'; assert module_path.exists()
if module_path not in sys.path: sys.path.append(str(module_path))

device_id = device_id if len(sys.argv)<2 else sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{device_id}')

wandb.init(project = project , name = name)


#__________________________________________________________ code _____________________________

# ___________________________________STAGE 1: pretraining _________________________________________

from STUPD_dataloaders.pprfcnDataset import pprfcnDataset
from STUPD_dataloaders.utils import split_dataset
from models.dynamic.pprfcn import PPRFCN
import torchvision.transforms as transforms

ds = pprfcnDataset(annotations_path = stupd_path/'annotations',
                        video_path = stupd_path/'stupd',
                         n_frames = n_frames, 
                         x_tfms = [transforms.ToPILImage("RGB"),
                                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                                  ],
                        )

train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =8 , shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = 16 , shuffle = True)

dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 3

model = PPRFCN(train_ds.c, pretrained = False).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))


learn.fit_one_cycle(5)
# learn.save(name)
# # ___________________________________STAGE 2: FINETUNING on spatialsense___________________________________



# from dataloaders.vidvrd.pprfcnDataset import pprfcnDataset
# from dataloaders.vidvrd.utils import  map_vidvrd_to_stupd


# train_ds = pprfcnDataset(annotations_directory_path = vidvrd_path/'train',
#                         video_path = vidvrd_path/'videos',
# #                          split='train',
#                          n_frames = n_frames, 
#                          x_tfms = [transforms.ToPILImage("RGB"),
#                                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
#                                   ],
#                          y_category_tfms = [map_vidvrd_to_stupd],
#                         )

# valid_ds = pprfcnDataset(annotations_directory_path = vidvrd_path/'test',
#                         video_path = vidvrd_path/'videos',
# #                          split='train',
#                          n_frames = n_frames,
#                          x_tfms = [transforms.ToPILImage("RGB")],
#                          y_category_tfms = [map_vidvrd_to_stupd])




# train_dl = DataLoader(train_ds, batch_size =16 , shuffle = True, num_workers = 0)
# valid_dl = DataLoader(valid_ds, batch_size = 32, shuffle = True, num_workers = 0)


# dls = DataLoaders(train_dl, valid_dl)
# dls.n_inp = 3

# model = PPRFCN(train_ds.c, pretrained = False).cuda()

# learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
#                 path = core_pth/'experiments/baselines/weights',
#                 model_dir = model_name,
#                 cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))


# learn.load(name, device = device)
# learn.fit(5)