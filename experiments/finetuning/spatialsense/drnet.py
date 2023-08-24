#____________________________________________WANDB info__________________________________________
#edit these
project = 'SpatialSense from scratch'
name = 'drnet-scratch'
model_name = "drnet"
dataset_name = 'SpatialSense'
#___________________________________________GPU info_____________________________________________

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


from dataloaders.spatialsense.drnetDataset import drnetDataset
from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd
from models.static.drnet import DRNet

import torchvision.transforms as transforms




#___________________________________________code _________________________________________________


train_ds = drnetDataset(annotations_path = spatialsenses_pth/'annotations.json',
                        image_path = spatialsenses_pth/'images',
                        encoder_path = encoder_path,
                        split = 'train',
                        y_category_tfms = [map_spatialsenses_to_stupd],
                        x_img_tfms =     [transforms.ToPILImage("RGB"),
                                        transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                                        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)],
                        bbox_mask_tfms = [transforms.ToPILImage("RGB"),
                                            transforms.Pad(4, padding_mode="edge"),
                                            transforms.RandomResizedCrop(32, scale=(0.75, 0.85))]
                         )

valid_ds = drnetDataset(annotations_path = spatialsenses_pth/'annotations.json',
                        image_path = spatialsenses_pth/'images',
                        encoder_path = encoder_path, 
                        split = 'valid',
                        y_category_tfms = [map_spatialsenses_to_stupd],
                        x_img_tfms =     [transforms.ToPILImage("RGB"),
                                            transforms.CenterCrop(224)],
                        
                        bbox_mask_tfms = [transforms.ToPILImage("RGB"),
                                            transforms.Pad(4, padding_mode="edge"),
                                            transforms.CenterCrop(32)]
                         )


train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, num_workers = 0)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, num_workers = 0)

dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 4

model = DRNet(word_embedding_dim = 300, 
              feature_dim = 512, 
              num_classes = train_ds.c, 
              num_layers = 3,
              imagenet_pretrained = False).cuda()

learn = Learner(dls, model = model, 
                loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.fit(5)