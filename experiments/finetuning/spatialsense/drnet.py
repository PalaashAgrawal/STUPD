#____________________________________________WANDB info__________________________________________
#edit these
project = 'SpatialSense_scratch_STUPDpretrained'
name = 'drnet-scratch-STUPDpretrained'
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
import torchvision.transforms as transforms
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





#___________________________________________code _________________________________________________

# ___________________________________STAGE 1: pretraining _________________________________________
from pretraining_dataloaders.stupd.spatialsense.drnetDataset import drnetDataset
from pretraining_dataloaders.stupd.spatialsense.utils import split_dataset
from models.static.drnet import DRNet


# ds = languageOnlyDataset(stupd_path/'annotations', encoder_path)
ds = drnetDataset(stupd_path/'annotations', stupd_path/'stupd', encoder_path,
                    x_img_tfms =        [transforms.ToPILImage("RGB"),
                                        transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                                        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)],

                    bbox_mask_tfms =    [transforms.ToPILImage("RGB"),
                                        transforms.Pad(4, padding_mode="edge"),
                                        transforms.RandomResizedCrop(32, scale=(0.75, 0.85))])

train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)

dls = DataLoaders(train_dl, valid_dl)
model = DRNet(word_embedding_dim = 300, 
              feature_dim = 512, 
              num_classes = train_ds.c, 
              num_layers = 3,
              imagenet_pretrained = False).cuda()

# model = SimpleLanguageOnlyModel(word_embedding_dim=300, feature_dim=512, c=train_ds.c).cuda()
learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
               path = core_pth/'experiments/baselines/weights',
               model_dir = model_name,
               )

learn.fit_one_cycle(2)
learn.save(name)

# ___________________________________STAGE 2: FINETUNING on spatialsense___________________________________



from dataloaders.spatialsense.drnetDataset import drnetDataset
from dataloaders.spatialsense.utils import  map_spatialsenses_to_stupd


import torchvision.transforms as transforms

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
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))

learn.load(name, device = device)

learn.fit(5)