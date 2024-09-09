import torch
import clip
from tqdm import tqdm
from PIL import Image

import pandas as pd
import os

from pathlib import Path
import open_clip

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch import nn


from fastai.losses import BaseLoss


#data
#_______________________________________________________________________________________________________________________    
#construction of dataset
dataset_parent_path = Path('/data/dataset/agrawalp2/stupd/stupd_dataset/stupd')
#list all folders in the dataset
annotations_path =  Path('/data/dataset/agrawalp2/stupd/stupd_dataset/annotations/')
#list all csv files in the annotation path
annotation_files = list(annotations_path.glob('*.csv'))


# Static prepositions
static_prepositions = ['above.csv', 'on.csv', 'along_position.csv', 'all_over.csv', 'among.csv', 'against_leaning.csv', 
                       'outside.csv', 'in_front_of.csv', 'behind.csv', 'between.csv', 'inside.csv', 'below.csv', 
                       'around_static.csv', 'beside.csv']

# Dynamic prepositions
dynamic_prepositions = ['around.csv', 'onto.csv', 'out_of.csv', 'up.csv', 'along.csv',
                        'into.csv', 'towards.csv', 'through.csv', 'from.csv', 'with.csv', 'against.csv', 'by.csv',
                        'off.csv', 'into_crash.csv', 'down.csv', 'over.csv']


all_prepositions = {i:k for k,i in enumerate(static_prepositions + dynamic_prepositions)}

def convert_to_valid_english(preposition):
    "Convert all prepositions into valid English descriptions. Eg. 'along_position' to 'along position'"

    if preposition == 'along_position.csv': return 'placed along'
    elif preposition == 'all_over.csv': return 'scattered all over the place'
    elif preposition == 'against_leaning.csv': return 'leaning against'
    elif preposition == 'around_static.csv': return 'placed around'
    elif preposition == 'above.csv': return 'above'
    elif preposition == 'on.csv': return 'on top of'
    elif preposition == 'among.csv': return 'placed among'
    elif preposition == 'outside.csv': return 'outside'
    elif preposition == 'in_front_of.csv': return 'in front of'
    elif preposition == 'behind.csv': return 'behind'
    elif preposition == 'between.csv': return 'placed between two'
    elif preposition == 'inside.csv': return 'placed inside'
    elif preposition == 'below.csv': return 'below'
    elif preposition == 'beside.csv': return 'beside'
    elif preposition == 'around.csv': return 'moving around'
    elif preposition == 'onto.csv': return 'moving onto'
    elif preposition == 'out_of.csv': return 'moving out of'
    elif preposition == 'up.csv': return 'moving up'
    elif preposition == 'along.csv': return 'moving along'
    elif preposition == 'into.csv': return 'moving into'
    elif preposition == 'towards.csv': return 'moving towards'
    elif preposition == 'through.csv': return 'moving through'
    elif preposition == 'from.csv': return 'moving away from'
    elif preposition == 'with.csv': return 'moving along with'
    elif preposition == 'against.csv': return 'moving against'
    elif preposition == 'by.csv': return 'flying by'
    elif preposition == 'off.csv': return 'flying off'
    elif preposition == 'into_crash.csv': return 'crashing into'
    elif preposition == 'down.csv': return 'moving down'
    elif preposition == 'over.csv': return 'moving over'
    
    
    else: return preposition.replace('_', ' ').replace('.csv', '')



#for each annotation csv file, there are multiple columns
#here is how images and text will be constructed
#image path is stored in column "image_path" as a list (eg ['above/above_0.png']). 
#extract this image path and construct the full path from the dataset_parent_path
#text is to be constructed as subject_category + relation + object_category


def construct_text_combinations(subject, object, preposition):
    # Construct text combinations for a given subject, object, and preposition
    valid_english_preposition = convert_to_valid_english(preposition)
    text_combination = f"{subject} {valid_english_preposition} {object}"
    
    #some prepositions involve multiple instances of the object. We would need to add an 's' to the object
    if preposition in ['among.csv', 'between.csv']: text_combination += 's'
        
    return text_combination

def construct_all_text_combinations(subject, object):
    # Construct all possible text combinations for a given subject and object, with all prepositions
    all_text_combinations = []
    for preposition in all_prepositions:
        text_combination = construct_text_combinations(subject, object, preposition)
            
        all_text_combinations.append(text_combination)
    return all_text_combinations

    
def process_csv_file(file_path, dataset_parent_path, n_frames = 3):
    '''
    Takes in a CSV file and returns a list of image paths, texts, and target labels constructed from the CSV file
    image_paths and target_labels are required for training, while we will use the candidate_labels for evaluation
    
    
    n_frames: int, default 1
    There are two kinds of inputs in STUPD: videos and images. For videos, we will uniformly sample frames from the video. For images, we will copy the image path number_of_frames_ton_consider times.
    '''
    
    
    #assert number_of_frames_to_consider is odd, so that symmetry is maintained
    assert n_frames%2 == 1, "number_of_frames_to_consider should be odd, for symmetric sampling"
    
    
    
    df = pd.read_csv(file_path)
    
    # Ensure required columns are present
    required_columns = ["image_path", "subject_category", "relation", "object_category"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"CSV file {file_path} is missing one or more required columns: {required_columns}")
    
    image_paths = []
    # candidate_labels = []
    target_labels = [] 
    
    for _, row in df.iterrows():
        # Extract and construct the full image path
        # image_path = os.path.join(dataset_parent_path, eval(row["image_path"])[0])
        
        #uniformly sample row["image_path"]to get number_of_frames_ton_consider frames, including first and last frame
        img_pth_list = eval(row["image_path"])
        images = (img_pth_list[::len(img_pth_list)//(n_frames-1)]+[img_pth_list[-1]]) if len(img_pth_list) > 1 else [img_pth_list[0]]*n_frames
        image_paths.append([os.path.join(dataset_parent_path, image) for image in images])
        
        # Construct the candidate labels text
        # candidate_labels.append(construct_all_text_combinations(row["subject_category"], row["object_category"]))
        target_labels.append(construct_text_combinations(row["subject_category"], row["object_category"], file_path.name))
    
    return image_paths, target_labels #, candidate_labels

#_______________________________________________________________________________________________________________________    





#_______________________________________________________________________________________________________________________    
#model
def zero_shot_classification(image_paths, target_label, candidate_labels):
    f"""
    calculate top1, top3, top5 accuracy of a given static csv file. 
    WILL HAVE TO REWRITE
    
    
    
    
    RIGHT NOW, WORKS FOR ONLY STATIC CSV FILES. 
    for all images (list) in a static class, this function computes the similarity between the image and all 
    candidate labels (30 preposition triplets in this case) (list of lists),
    and calculates accuracy (top-1, top-3, and top-5) based on the target label (sigle string).
    """
    
    results = []
    for image_path, labels in tqdm(zip(image_paths, candidate_labels)):
        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Preprocess the candidate labels
        # text_inputs = torch.cat([clip.tokenize(f"{c}") for c in labels]).to(device)
        text_inputs = tokenizer(labels).to(device)
        
        # Encode the image and labels
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)
        
        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute the similarity between the image and labels
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get the indices of the top 1, top 3, and top 5 labels
        sorted_indices = similarity.argsort(dim=-1, descending=True)
        top1_indices = sorted_indices[:, 0].tolist()
        top3_indices = sorted_indices[:, :3].tolist()
        top5_indices = sorted_indices[:, :5].tolist()
        
        # Convert indices to labels
        top1_labels = [labels[i] for i in top1_indices]
        top3_labels = [labels[i] for i in top3_indices[0]]
        top5_labels = [labels[i] for i in top5_indices[0]]
        
        # Check if the target label is in the top 1, top 3, and top 5 labels
        # top1_result = int(target_label in top1_labels)
        top1_result = int(any(target_label in label for label in top1_labels))
        top3_result = int(any(target_label in label for label in top3_labels))
        top5_result = int(any(target_label in label for label in top5_labels))

        results.append((top1_result, top3_result, top5_result))
    
    #get aggregate top1, top3 and top5 accuracy
    top1_accuracy = sum(result[0] for result in results) / len(results)
    top3_accuracy = sum(result[1] for result in results) / len(results)
    top5_accuracy = sum(result[2] for result in results) / len(results)
    
    return top1_accuracy, top3_accuracy, top5_accuracy
    



class STUPD_VLdata_Train(Dataset):
    def __init__(self, preposition_to_consider, annotations_path, dataset_parent_path,preprocess_fn, tokenizer, device = None,  n_frames = 3 ):
        f"""
        we will consider multiple frames for STUPD. For static, repeat the frames. For dynamic, sample uniformly.
        """
        self.n_frames = n_frames
        self.data = []
        self.device = device
        
        
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer
        for preposition_file in preposition_to_consider:
            csv_file = annotations_path / preposition_file
            image_paths, target_labels = process_csv_file(csv_file, dataset_parent_path)
            for image_path, target_label in zip(image_paths, target_labels):
                self.data.append((image_path, target_label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, target_label = self.data[idx]
        # Load the image (assuming a function load_image exists)
        # image = preprocess(Image.open(image_path)).to(device) #3xHxW
        image = torch.cat([self.preprocess(Image.open(img)) for img in image_path], dim = 0).to(self.device or 'cuda') #(3*n_frames)xHxW
        target_label = self.tokenizer(target_label).squeeze().to(self.device or 'cuda') #(,77) , 77 is the max token limit for CLIP
        # candidate_labels = tokenizer(candidate_labels).to(device) #30x77, used only for classification (among 30 labels) in evaluation,
        #we dont need all the 30 labels for STUPD, we will need them only for zero shot in evaluation for real world datasets. 
        
        return image, target_label, target_label #fastai expects a target tensor, so we return None for now.
    
#dataloader
def get_dataloader(preposition_to_consider, annotations_path, dataset_parent_path, batch_size=32, shuffle=True, **kwargs):
    dataset = STUPD_VLdata_Train(preposition_to_consider, annotations_path, dataset_parent_path, **kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


class CLIP_loss(BaseLoss):
    def __init__(self):
        from open_clip import ClipLoss
        self.func = ClipLoss
        self.loss = ClipLoss()
        
    def __call__(self, pred, target = None):
        a,b,c = pred
        return self.loss(a,b,c,output_dict = False)
    
    

#customizing the model architecture. 


    
def modify_model_architecture(model, n_frames = 3):
    device = next(model.parameters()).device
    model.visual.conv1 = torch.nn.Conv2d(3*n_frames, 768,kernel_size=(32, 32), stride=(32, 32), bias=False).to(device)
    return model
    
    


