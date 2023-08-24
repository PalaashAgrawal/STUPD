from gensim.models import KeyedVectors
from torch.utils.data import Dataset

import numpy as np
from autocorrect import Speller
from pathlib import Path
import os
import cv2
import torch
from torch import nn

# spatialsenses_to_stupd = {
#     "above": "above",
#     "behind": "behind",
#     "in": "inside",
#     "in front of": "in_front_of",
#     "next to": "beside",
#     "on": "on",
#     "to the left of": "beside",
#     "to the right of": "beside",
#     "under": "below",
# }

stupd_classes = set(['above', 'below',
                    'inside', 'beside',
                    'in_front_of', 'behind', 'on'])



class word2vec():
    f'''
    This function converts a word (string) into a vector of length word_embedding_dim through a word2vec function. 
    At max, only max_phrase_len number of words are chosen from the input string. If max_phrase_len is None, then all the words are selected. 
    '''
    def __init__(self, encoder_path, max_phrase_len: int = None , word_embedding_dim: int = 300, spell_func = Speller):

        assert Path(encoder_path).exists()
        self.word2vec_encoder = KeyedVectors.load_word2vec_format(Path(encoder_path), binary=True, unicode_errors="ignore")

        self.max_phrase_len = max_phrase_len
        self.word_embedding_dim = word_embedding_dim

        self._spell = spell_func()

    def __call__(self, phrase):
        max_phrase_len = self.max_phrase_len or len(phrase.split())
        vec = np.zeros((max_phrase_len, self.word_embedding_dim), dtype=np.float32)
        for i, word in enumerate(phrase.split()[:max_phrase_len]):
            if word in self.word2vec_encoder: vec[i] = self.word2vec_encoder[word]
            elif self._spell(word) in self.word2vec_encoder: vec[i] = self.word2vec_encoder[self._spell(word)]
            else: pass
        return torch.from_numpy(vec)



def read_img(url, imagepath):
    if url.startswith("http"):  # flickr
        filename = os.path.join(imagepath, "flickr", url.split("/")[-1])
    else:  # nyu
        filename = os.path.join(imagepath, "nyu", url.split("/")[-1])
    return filename


def split_dataset(dataset, pct = 0.8):
    f'''
    split a dataset into train and validation dataset, with train getting pct % of data. 
    '''
    assert 0.<=pct<=1., f'pct should be between 0 and 1'
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [int(len(dataset)*pct), len(dataset) - int(len(dataset)*pct)])
    train_ds.c, valid_ds.c = dataset.c, dataset.c
    return train_ds, valid_ds 


def convert_stupdBbox_to_spatialSenseBbox(bbox):
    f'''
    Our models expect a standard input format for bounding boxes, i.e the SpatialSense format. 
    #stupd bbox format = wmin, hmin, w, h
    #spatialsenses bbox format = hmin, hmax, wmin, wmax

    '''
    
    wmin, hmin, w,h = bbox
    return (hmin, hmin+h, wmin, wmin+w)