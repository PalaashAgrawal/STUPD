import torch
from torch import nn
import torch.nn.functional as F


#_____________________________________________________________________________________________
# ... model description ...

# subj (word)
#             \
#             Embedding(300)
#                             \   
#                              concat(600) --> linear -->linear --> predicate_category
#                             /
#             Embedding(300)
#             /
# obj (word)
#_____________________________________________________________________________________________


class PhraseEncoder(nn.Module):
    f'''
    This function converts the vectors from 2 or more words (of length word_embedding_dim each) into one single vector of length word_embedding_dim
    '''
    def __init__(self, word_embedding_dim, num_layers = 1, batch_first = True, bidirectional = True):
        super().__init__()
        self.encoder = nn.GRU(input_size = word_embedding_dim, 
                                     hidden_size = word_embedding_dim//2,
                                     num_layers = num_layers,
                                     batch_first = batch_first,
                                     bidirectional = bidirectional,
                                    )

    def forward(self, x): 
        
        out = self.encoder(x)[0]
        return torch.squeeze(out[:,-1,:])

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=True, relu=True):
        super().__init__()
        
        self.bn = bn
        self.relu = relu
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        if self.bn: x = self.batchnorm(x)
        if self.relu: x = self.ReLU(x)
        return x
    
    
class SimpleLanguageOnlyModel(nn.Module):
    def __init__(self, word_embedding_dim, feature_dim, c):
        super().__init__()
        self.phrase_encoder = PhraseEncoder(word_embedding_dim)

        self.linear1 = LinearBlock(word_embedding_dim, feature_dim)
        self.linear2 = LinearBlock(word_embedding_dim, feature_dim)
        self.linear3 = LinearBlock(2*feature_dim, feature_dim)
        self.linear4 = nn.Linear(feature_dim, c)
        
    def forward(self, subj, obj):
        # subj = torch.squeeze(self.phrase_encoder(subj)[0][:,-1,:])
        # obj = torch.squeeze(self.phrase_encoder(obj)[0][:,-1,:])
        
        subj = self.linear1(self.phrase_encoder(subj))
        obj  = self.linear2(self.phrase_encoder(obj))
        
        x = torch.cat((subj,obj), dim = 1)
        x = self.linear4(self.linear3(x))
        return x