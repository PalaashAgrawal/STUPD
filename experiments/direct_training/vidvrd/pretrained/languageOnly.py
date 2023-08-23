
#____________________________________________WANDB info__________________________________________
#edit these
project = 'VidVRD from scratch'
name = 'language-only'
model_name = "custom language based"
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

module_path = os.path.abspath(os.path.join('../../../baselines'))
if module_path not in sys.path: sys.path.append(module_path)

device_id = device_id if len(sys.argv)<2 else sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{device_id}')
wandb.init(project = project , name = name)

from dataloaders.vidvrd.languageOnlyDataset import languageOnlyDataset
from dataloaders.vidvrd.utils import  map_vidvrd_to_stupd
from models.dynamic.language_only import SimpleLanguageOnlyModel


#___________________________________________code _________________________________________________



train_ds = languageOnlyDataset(vidvrd_path/'train',
                              encoder_path = encoder_path,
                              y_tfms = [map_vidvrd_to_stupd])

valid_ds = languageOnlyDataset(vidvrd_path/'test',
                              encoder_path = encoder_path, 
                              y_tfms = [map_vidvrd_to_stupd])


train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)


dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 2
model = SimpleLanguageOnlyModel(word_embedding_dim=300, feature_dim=512, c=train_ds.c).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.fit(5)