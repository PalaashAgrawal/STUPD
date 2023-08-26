#____________________________________________WANDB info__________________________________________
#edit these
project = 'VidVRDscratch_stupdPretrained'
name = 'drnet'
model_name = "drnet"
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
stupd_path = Path('/data/dataset/agrawalp2/stupd/stupd_dataset'); assert stupd_path.exists()


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
from pretraining_dataloaders.stupd.vidvrd.drnetDataset import drnetDataset
from pretraining_dataloaders.stupd.vidvrd.utils import split_dataset
from models.dynamic.drnet import DRNet
import torchvision.transforms as transforms



ds = drnetDataset(annotations_path = stupd_path/'annotations',
                    video_path = stupd_path/'stupd',
                    encoder_path = encoder_path,
                    n_frames = n_frames,
                    x_img_tfms =     [transforms.ToPILImage("RGB"),
                                    transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)],
                    
                    bbox_mask_tfms = [transforms.ToPILImage("RGB"),
                                        transforms.Pad(4, padding_mode="edge"),
                                        transforms.RandomResizedCrop(32, scale=(0.75, 0.85))]
                         )
train_ds, valid_ds = split_dataset(ds, pct = 0.8)

train_dl = DataLoader(train_ds, batch_size =64 , shuffle = True, drop_last = True)
valid_dl = DataLoader(valid_ds, batch_size = 128 , shuffle = True, drop_last = True)
dls = DataLoaders(train_dl, valid_dl)
dls.n_inp = 4

model = DRNet(word_embedding_dim = 300, 
              feature_dim = 512, 
              n_frames = n_frames,
              num_classes = train_ds.c, 
              num_layers = 3,
              pretrained = False).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,
                )

learn.fit_one_cycle(2)
learn.save(name)

# ___________________________________STAGE 2: FINETUNING on spatialsense___________________________________

from dataloaders.vidvrd.drnetDataset import drnetDataset
from dataloaders.vidvrd.utils import  map_vidvrd_to_stupd




train_ds = drnetDataset(annotations_directory_path = vidvrd_path/'train',
                        video_path = vidvrd_path/'videos',
                        encoder_path = encoder_path,
                        n_frames = n_frames,
                        y_category_tfms = [map_vidvrd_to_stupd],
                        x_img_tfms =     [transforms.ToPILImage("RGB"),
                                        transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                                        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)],
                        
                        bbox_mask_tfms = [transforms.ToPILImage("RGB"),
                                            transforms.Pad(4, padding_mode="edge"),
                                            transforms.RandomResizedCrop(32, scale=(0.75, 0.85))]
                         )

valid_ds = drnetDataset(annotations_directory_path = vidvrd_path/'test',
                        video_path = vidvrd_path/'videos',
                        encoder_path = encoder_path, 
                        n_frames = n_frames,
                        y_category_tfms = [map_vidvrd_to_stupd],
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
              n_frames = n_frames,
              num_classes = train_ds.c, 
              num_layers = 3,
              pretrained = False).cuda()

learn = Learner(dls, model = model, loss_func = CrossEntropyLossFlat(), metrics = [accuracy,BalancedAccuracy()],
                path = core_pth/'experiments/baselines/weights',
                model_dir = model_name,
                cbs = WandbCallback (model_name = model_name , dataset_name = dataset_name))


learn.load(name, device = device)
learn.fit(5)