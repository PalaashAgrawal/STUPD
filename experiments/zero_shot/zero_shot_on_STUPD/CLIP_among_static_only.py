#restrict candidate labels to static only. The results otherwise (with 30 candidates) are terrible.
# Hypothesis: we are anyways not providing any dynamic info, so labels like "moving around" may be confusing the model. 
#modification: change the definition of all_prepositions

import torch
import clip
from tqdm import tqdm
from PIL import Image

import pandas as pd
import os

from pathlib import Path
import logging



# Load the CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) #VITs have best performance


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


all_prepositions = {i:k for k,i in enumerate(static_prepositions)}

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




def construct_all_text_combinations(subject, object):
    # Construct all possible text combinations for a given subject and object, with all prepositions
    all_text_combinations = []
    for preposition in all_prepositions:
        valid_english_preposition = convert_to_valid_english(preposition)
        text_combination = f"{subject} {valid_english_preposition} {object}"
        
        #some prepositions involve multiple instances of the object. We would need to add an 's' to the object
        if preposition in ['among.csv', 'between.csv']: text_combination += 's'
            
        all_text_combinations.append(text_combination)
    return all_text_combinations

    
def process_csv_file(file_path, dataset_parent_path):
    '''
    Takes in a CSV file and returns a list of image paths, texts, and target labels constructed from the CSV file
    '''
    df = pd.read_csv(file_path)
    
    # Ensure required columns are present
    required_columns = ["image_path", "subject_category", "relation", "object_category"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"CSV file {file_path} is missing one or more required columns: {required_columns}")
    
    image_paths = []
    candidate_labels = []
    target_label = convert_to_valid_english(file_path.name)
    
    for _, row in df.iterrows():
        # Extract and construct the full image path
        image_path = os.path.join(dataset_parent_path, eval(row["image_path"])[0])
        image_paths.append(image_path)
        
        # Construct the candidate labels text
        candidate_labels.append(construct_all_text_combinations(row["subject_category"], row["object_category"]))

    
    return image_paths, candidate_labels, target_label



def zero_shot_classification(image_paths, candidate_labels, target_label):
    results = []
    for image_path, labels in (zip(image_paths, candidate_labels)):
        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Preprocess the candidate labels
        # text_inputs = torch.cat([clip.tokenize(f"{c}") for c in labels]).to(device)
        text_inputs = torch.cat(clip.tokenize(*labels)).to(device)
        
        
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
        break
    
    #get aggregate top1, top3 and top5 accuracy
    top1_accuracy = sum(result[0] for result in results) / len(results)
    top3_accuracy = sum(result[1] for result in results) / len(results)
    top5_accuracy = sum(result[2] for result in results) / len(results)
    
    return top1_accuracy, top3_accuracy, top5_accuracy
    



# zero_shot_classification(*process_csv_file(annotations_path/static_prepositions[0], dataset_parent_path))  
# Run the above function for all static preposition csv files in the annotation path
results = {}
for preposition_file in static_prepositions:
    print(f"Processing {preposition_file}")
    csv_file = annotations_path / preposition_file
    image_paths, candidate_labels, target_label = process_csv_file(csv_file, dataset_parent_path)
    result = zero_shot_classification(image_paths, candidate_labels, target_label)
    
    results[target_label] = result
    print(f"Top-1 Accuracy: {result[0]:.2f}, Top-3 Accuracy: {result[1]:.2f}, Top-5 Accuracy: {result[2]:.2f}")
    