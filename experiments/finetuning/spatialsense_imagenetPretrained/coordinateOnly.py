

#____________________________________________WANDB info__________________________________________

#edit these
#edit these
project = 'SpatialSense_ImagenetPretrained_STUPDfinetuned'
name = 'coordinateOnly-pretrained-STUPDpretrained'
model_name = "CoordinateOnly"
dataset_name = 'SpatialSense'
#___________________________________________GPU info________
#___________________________________________GPU info_____________________________________________

#edit
device_id = 0 #cuda ID for the GPU, if it exists. If you dont have GPU, ignore this.

#___________________________________________path info______________________________________________
from pathlib import Path
core_pth = Path('/home/agrawalp2/prepositions'); assert core_pth.exists() 
spatialsenses_pth = core_pth/Path('real_world_data/spatialsense'); assert spatialsenses_pth.exists()
encoder_path = core_pth/Path('experiments/baselines/models/encoder/GoogleNews-vectors-negative300.bin.gz'); assert encoder_path.exists()

stupd_path = Path('/data/dataset/agrawalp2/stupd/stupd_dataset'); assert stupd_path.exists()
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


# ___________________________________STAGE 1: pretraining _________________________________________
from pretraining_dataloaders.stupd.spatialsense.coordinate2dOnlyDataset import coordinate2D_OnlyDataset
from pretraining_dataloaders.stupd.spatialsense.utils import split_dataset
from models.static.coordinate_2d_only import coordinateOnlyModel


# ds = languageOnlyDataset(stupd_path/'annotations', encoder_path)
ds = coordinate2D_OnlyDataset(stupd_path/'annotations')
train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)

dls = DataLoaders(train_dl, valid_dl)
model = coordinateOnlyModel(10, 64, train_ds.c).cuda()

# model = SimpleLanguageOnlyModel(word_embedding_dim=300, feature_dim=512, c=train_ds.c).cuda()
learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
               path = core_pth/'experiments/baselines/weights',
               model_dir = model_name,
               )

learn.fit_one_cycle(2)
learn.save(name)



# ___________________________________STAGE 2: FINETUNING on spatialsense___________________________________


from dataloaders.spatialsense.coordinate2dOnlyDataset import coordinate2D_OnlyDataset
from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd


#__________________________________________________________ code _____________________________

train_ds = coordinate2D_OnlyDataset(annotations_path = spatialsenses_pth/'annotations.json',
                                    split = 'train',
                                    x_tfms = None, 
                                    y_tfms = [map_spatialsenses_to_stupd])

valid_ds = coordinate2D_OnlyDataset(annotations_path = spatialsenses_pth/'annotations.json',
                                    split = 'valid',
                                    x_tfms = None, 
                                    y_tfms = [map_spatialsenses_to_stupd])


train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)


dls = DataLoaders(train_dl, valid_dl)
model = coordinateOnlyModel(10, 64, train_ds.c).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))


learn.load(name, device = device)
learn.fit(5)

