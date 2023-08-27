

#____________________________________________WANDB info__________________________________________

#edit these
project = 'directly on STUPD'
name = 'coordinate-only'
model_name = "custom language based"
dataset_name = 'ViDVRD'

#___________________________________________GPU info_____________________________________________

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
from STUPD_dataloaders.coordinate2dOnlyDataset import coordinate2D_OnlyDataset
from STUPD_dataloaders.utils import split_dataset
from models.dynamic.coordinate_2d_only import coordinateOnlyModel


# ds = languageOnlyDataset(stupd_path/'annotations', encoder_path)
ds = coordinate2D_OnlyDataset(stupd_path/'annotations',
                              n_frames = n_frames)

train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)

dls = DataLoaders(train_dl, valid_dl)
model = coordinateOnlyModel(10*n_frames, 64, train_ds.c).cuda()

# model = SimpleLanguageOnlyModel(word_embedding_dim=300, feature_dim=512, c=train_ds.c).cuda()
learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
               path = core_pth/'experiments/baselines/weights',
               model_dir = model_name,
               cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name)
               )

learn.fit_one_cycle(5)
learn.save(name)

# # ___________________________________STAGE 2: FINETUNING on spatialsense___________________________________



# from dataloaders.vidvrd.coordinate2dOnlyDataset import coordinate2D_OnlyDataset
# from dataloaders.vidvrd.utils import  map_vidvrd_to_stupd



# train_ds = coordinate2D_OnlyDataset(annotations_directory_path = vidvrd_path/'train',
#                                     n_frames = n_frames,
#                                     x_tfms = None, 
#                                     y_tfms = [map_vidvrd_to_stupd])

# valid_ds = coordinate2D_OnlyDataset(annotations_directory_path = vidvrd_path/'test',
#                                     n_frames = n_frames,
#                                     x_tfms = None, 
#                                     y_tfms = [map_vidvrd_to_stupd])


# train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
# valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)


# dls = DataLoaders(train_dl, valid_dl)
# model = coordinateOnlyModel(10*n_frames, 64, train_ds.c).cuda()

# learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
#                 path = core_pth/'experiments/baselines/weights',
#                 model_dir = model_name,
#                 cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

# learn.load(name, device = device)
# learn.fit(5)

