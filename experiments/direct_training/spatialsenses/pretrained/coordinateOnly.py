

#____________________________________________WANDB info__________________________________________

#edit these
project = 'SpatialSense pretrained'
name = 'coordinate-only-pretrained'
model_name = "custom language based"
dataset_name = 'SpatialSense'

#___________________________________________GPU info_____________________________________________

#edit
device_id = 0 #cuda ID for the GPU, if it exists. If you dont have GPU, ignore this.

#___________________________________________path info______________________________________________
from pathlib import Path

#edit. Expected structure: core_path contains dataset path as well as the code for the experiments. 
core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists() 
spatialsenses_pth = core_pth/Path('real_world_data/spatialsense'); assert spatialsenses_pth.exists()
encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()

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


from dataloaders.spatialsense.coordinate2dOnlyDataset import coordinate2D_OnlyDataset
from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd
from models.static.coordinate_2d_only import coordinateOnlyModel


#__________________________________________________________ code _____________________________

train_ds = coordinate2D_OnlyDataset(annotations_path = spatialsenses_pth/'annotations.json',
                                    split = 'train',
                                    x_tfms = None, 
                                    y_tfms = [map_spatialsenses_to_stupd])

valid_ds = coordinate2D_OnlyDataset(annotations_path = spatialsenses_pth/'annotations.json',
                                    split = 'valid',
                                    x_tfms = None, 
                                    y_tfms = [map_spatialsenses_to_stupd])

len(train_ds),len(valid_ds)

train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)


dls = DataLoaders(train_dl, valid_dl)
model = coordinateOnlyModel(10, 64, train_ds.c).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.fit(5)

