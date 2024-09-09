



#training scrpt




from CLIP_for_STUPD import modify_model_architecture, get_dataloader,CLIP_loss\
    ,all_prepositions, annotations_path, dataset_parent_path, static_prepositions
import torch
import open_clip
from fastai.vision.all import *
from fastai.callback.wandb import *
import wandb




log_wandb = True #set to False if you dont want to log progress to W&B

project = 'STUPD_CLIP'
dataset = "STUPD"
mode = 'scratch'
model= 'CLIP ViT-B-32'
# ______________________


cbs = []
if log_wandb:
    cbs.append(WandbCallback())
    wandb.init(project=project, name = f"{model}_{dataset}")





n_frames = 3 #3 frames for each preposition example






# Load the CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device) #VITs have best performance
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)    

model = modify_model_architecture(model, n_frames = n_frames)




tokenizer = open_clip.get_tokenizer('ViT-B-32')



dl = get_dataloader(all_prepositions, annotations_path, dataset_parent_path,  batch_size=32, preprocess_fn = preprocess, tokenizer = tokenizer, device= device)
dls = DataLoaders(dl, dl)
dls.n_inp = 2   




class CustomLearner(Learner):
    #skipping validation 
    
    
     def _do_epoch(self):
        self._do_epoch_train()



learn = CustomLearner(dls, model, loss_func=CLIP_loss(), cbs =cbs)
learn.fit(1)


#save the model
learn.save('STUPD_pretrained_Video_CLIP')

