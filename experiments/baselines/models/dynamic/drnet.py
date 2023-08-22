import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models.video as models
from collections import OrderedDict


#helper models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.layernorm1 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels // 2, kernel_size=3, padding=1
        )
        self.layernorm2 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inp):
        x = F.relu(self.layernorm1(self.conv1(inp)))
        x = F.relu(self.layernorm2(self.conv2(x)))
        x = self.conv3(x)
        return x + self.conv_skip(inp)



class Hourglass(nn.Module):
    def __init__(self, im_size, feature_dim):
        super().__init__()
        assert im_size == 1 or im_size % 2 == 0
        self.skip_resblock = ResidualBlock(feature_dim, feature_dim, im_size)
        if im_size > 1:
            self.pre_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)
            self.layernorm1 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.sub_hourglass = Hourglass(im_size // 2, feature_dim)
            self.layernorm2 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.post_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)

    def forward(self, x):
        up = self.skip_resblock(x)
        if x.size(-1) == 1:
            return up
        down = F.max_pool2d(x, 2)
        down = F.relu(self.layernorm1(self.pre_resblock(down)))
        down = F.relu(self.layernorm2(self.sub_hourglass(down)))
        down = self.post_resblock(down)
        # down = F.upsample(down, scale_factor=2) #upsample is deprecated now. 
        down = F.interpolate(down, scale_factor=2) 
        return up + down   
    



class convert_3d_to_2d(nn.Module):
    def __init__(self):
        super().__init__(self)
    def forward(self, x):
        return x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])



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
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.ReLU(x)
        return x

class DRNet(nn.Module):
    def __init__(self, word_embedding_dim:int, feature_dim:int, n_frames:int, num_classes:int, num_layers=3, pretrained:bool = True):

        f'''
        word_embedding_dim: word2vec output length
        feature_dim: dimension of final fc layer
        n_frames: how many frames are used from a video. 
        num_classes: how many categories are there to classify from. 
        num_layers: intermediate number of layers in PhraseEncoder (which converts multiple word embeddings into one single mebedding)
        pretrained: whether or not self.appr_module backbone is pretrained. If True, Kinetics400_V1 weights are used for the 3d module
        '''

        super(DRNet, self).__init__()
        

        self.phrase_encoder =  PhraseEncoder(word_embedding_dim)
        self.feature_dim = feature_dim
        
        self.n_frames = n_frames


        weights = models.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.appr_module = models.__dict__['r3d_18'](weights=weights)


        self.appr_module.fc = nn.Linear(512, feature_dim//2)
        
        self.num_layers = num_layers
        self.c = num_classes

        self.pos_module = nn.Sequential(
            OrderedDict(
                [
                    ("conv1_p", nn.Conv2d(2*self.n_frames, 32, 5, 2, 2)),
                    # ("conv1_p", nn.Conv3d(3,1, 3)),
                    #PAg: added this below layer as well
                    # ("3d_to_2d", convert_3d_to_2d()), 
                    ("batchnorm1_p", nn.BatchNorm2d(32)),
                    ("relu1_p", nn.ReLU()),
                    ("conv2_p", nn.Conv2d(32, 64, 3, 1, 1)),
                    ("batchnorm2_p", nn.BatchNorm2d(64)),
                    ("relu2_p", nn.ReLU()),
                    ("maxpool2_p", nn.MaxPool2d(2)),
                    ("hg", Hourglass(8, 64)),
                    ("batchnorm_p", nn.BatchNorm2d(64)),
                    ("relu_p", nn.ReLU()),
                    ("maxpool_p", nn.MaxPool2d(2)),
                    ("conv3_p", nn.Conv2d(64, feature_dim//2, 4)),
                    ("batchnorm3_p", nn.BatchNorm2d(feature_dim//2)),
                ]
            )
        )

#         self.PhiR_0 = nn.Linear(512, feature_dim)
        self.batchnorm = nn.BatchNorm1d(feature_dim)

        self.Phi_subj = nn.Linear(300, feature_dim//2)
        self.Phi_obj = nn.Linear(300, feature_dim//2)

        self.fc = nn.Sequential(LinearBlock(2*feature_dim,feature_dim), 
                                LinearBlock(feature_dim, feature_dim//2), 
                                nn.Linear(feature_dim//2, self.c))

    def forward(self, subj, obj, im, posdata):



        appr_feature = self.appr_module(im) #output 256 features

        pos_feature = (self.pos_module(posdata).view(-1, self.feature_dim//2))#output 256 features
        
        
        qa = self.phrase_encoder(subj)
        qb = self.phrase_encoder(obj)
        
        x = torch.cat([appr_feature, 
                       pos_feature,
                       self.Phi_subj(qa), self.Phi_obj(qb),
                      ], 1)

        x = self.fc(x)

        return x