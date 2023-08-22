from gensim.models import KeyedVectors
from torch.utils.data import Dataset

import numpy as np
from autocorrect import Speller
from pathlib import Path
import os
import cv2
import torch
from torch import nn

vidvrd_to_stupd = {
    "behind": "behind",
    "front": "in_front_of",
    "away": "from",
    "with": "with",
    "left": "beside",
    "right": "beside",
    "next_to": "beside",
    "above": "above",
    "beneath": "below",
    "toward": "towards",
    "past": "by",
    "inside": "inside" #does removing this improve accuracy?
}



def map_vidvrd_to_stupd(o, mapping_dict = vidvrd_to_stupd):
    f'''
    This function maps predicates from vidvrd to predicates as seen in the stupd dataset. 
    If the predicate does not match any stupd predicate, then this function returns None
    '''
    # return mapping_dict.get(o)
    for relation in mapping_dict:
        if relation in o: return mapping_dict.get(o)
    
    return None



class word2vec():
    f'''
    This function converts a word (string) into a vector of length word_embedding_dim through a word2vec function. 
    At max, only max_phrase_len number of words are chosen from the input string. If max_phrase_len is None, then all the words are selected. 

    Note that in vidvrd, words are separated by '_' and not space. 
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
        for i, word in enumerate(phrase.split('_')[:max_phrase_len]):
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
