#____________________________________________WANDB info__________________________________________
#edit these
project = 'VidVRD pretrained'
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
core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists()
vidvrd_path = core_pth/Path('real_world_data/vidvrd/vidvrd-dataset'); assert vidvrd_path.exists()
encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()

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


device_id = device_id if len(sys.argv)<2 else sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{device_id}')

wandb.init(project = project , name = name)

module_path = os.path.abspath(os.path.join('../../../baselines'))
if module_path not in sys.path: sys.path.append(module_path)


from dataloaders.vidvrd.pprfcnDataset import pprfcnDataset
from dataloaders.vidvrd.utils import  map_vidvrd_to_stupd
from models.dynamic.pprfcn import PPRFCN
import torchvision.transforms as transforms
#___________________________________________code _________________________________________________

train_ds = pprfcnDataset(annotations_directory_path = vidvrd_path/'train',
                        video_path = vidvrd_path/'videos',
#                          split='train',
                         n_frames = n_frames, 
                         x_tfms = [transforms.ToPILImage("RGB"),
                                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                                  ],
                         y_category_tfms = [map_vidvrd_to_stupd],
                        )

valid_ds = pprfcnDataset(annotations_directory_path = vidvrd_path/'test',
                        video_path = vidvrd_path/'videos',
#                          split='train',
                         n_frames = n_frames,
                         x_tfms = [transforms.ToPILImage("RGB")],
                         y_category_tfms = [map_vidvrd_to_stupd])




train_dl = DataLoader(train_ds, batch_size =32 , shuffle = True, num_workers = 0)
valid_dl = DataLoader(valid_ds, batch_size = 64 , shuffle = True, num_workers = 0)


dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 3

model = PPRFCN(train_ds.c, pretrained = True).cuda()
learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],  cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.fit(5)