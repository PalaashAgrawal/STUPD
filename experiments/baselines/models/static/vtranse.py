import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
import math

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

class vtranse_utils(nn.Module):

    def __init__(self):
        super().__init__()


    def _fix_bbox(self, bbox, size):
        new_bbox = [
            int(bbox[0]),
            min(size, int(math.ceil(bbox[1]))),
            int(bbox[2]),
            min(size, int(math.ceil(bbox[3]))),
        ]

        assert (
            0 <= new_bbox[0] < new_bbox[1] <= size
            and 0 <= new_bbox[2] < new_bbox[3] <= size
        )
        return new_bbox
    



    
    def _normalize_bbox(self, bbox, img_shape):
        f'''
        PAg: ROI pooling functions (such as self._roi_pool scale bboxes to the shape of output matrix by multiplying. 
        If we don't divide by the original image size first, 
        then the multiplication will lead to extremely large values, greater than the size of the output matrix. 
        This will throw error. 

        Hence this function
        '''
        H,W = img_shape[-2], img_shape[-1]
        bbox=torch.div(bbox,torch.Tensor([H,H,W,W]).to(self.device))

        #PAg: Had to add this extra engineering logic, because in many cases, hmax/wmax is greater than the size of the image itself. 
        bbox = torch.div(bbox, torch.max(torch.max(bbox), torch.Tensor([1.]).to(self.device)))

        return bbox
        

    

        
class VtransE(vtranse_utils):
    def __init__(self,
                word_embedding_dim:int, #for phrase encoder. 
                num_classes:int,
                visual_feature_size:int = 3, #defaults according to SpatialSense implementation.
                predicate_embedding_dim:int = 512, # default according to SpatialSense implementation. 
                imagenet_pretrained:bool = True,
                ):

        super().__init__()

        self.visual_feature_size = visual_feature_size
        self.phrase_encoder =  PhraseEncoder(word_embedding_dim)
        self.c = num_classes
        self.device = None

        weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        self.backbone = models.__dict__['resnet18'](weights=weights)

        self.backbone = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        self.scale_factor = nn.Parameter(torch.Tensor(3))
        nn.init.uniform_(self.scale_factor)

        self.linear1 = nn.Linear(
            visual_feature_size * visual_feature_size * 512,
            visual_feature_size * visual_feature_size * 64,
        )
        self.batchnorm1 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)
        self.linear2 = nn.Linear(
            visual_feature_size * visual_feature_size * 512,
            visual_feature_size * visual_feature_size * 64,
        )
        self.batchnorm2 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)

        feature_dim = 300 + 4 + visual_feature_size * visual_feature_size * 64
        self.W_o = nn.Linear(feature_dim, predicate_embedding_dim)
        self.W_s = nn.Linear(feature_dim, predicate_embedding_dim)
        self.W_p = nn.Linear(predicate_embedding_dim, self.c)





    def forward(self, subj, obj, full_im, t_s, t_o, bbox_s, bbox_o):

        if self.device is None: self.device = next(self.parameters()).device

        classeme_subj = self.phrase_encoder(subj)
        classeme_obj = self.phrase_encoder(obj)

        img_feature_map = self.backbone(full_im)
        subj_img_feature = []
        obj_img_feature = []

        bbox_s, bbox_o = self._normalize_bbox(bbox_s, full_im.shape), self._normalize_bbox(bbox_o, full_im.shape)

        for i in range(bbox_s.size(0)):
            bbox_subj = self._fix_bbox(7 * bbox_s[i], 7)
            bbox_obj = self._fix_bbox(7 * bbox_o[i], 7)
            subj_img_feature.append(
                F.interpolate(
                    img_feature_map[
                        i : (i + 1),
                        :,
                        bbox_subj[0] : bbox_subj[1],
                        bbox_subj[2] : bbox_subj[3],
                    ],
                    self.visual_feature_size,
                    mode="bilinear",
                )
            )
            obj_img_feature.append(
                F.interpolate(
                    img_feature_map[
                        i : (i + 1),
                        :,
                        bbox_obj[0] : bbox_obj[1],
                        bbox_obj[2] : bbox_obj[3],
                    ],
                    self.visual_feature_size,
                    mode="bilinear",
                )
            )
        subj_img_feature = torch.cat(subj_img_feature)
        obj_img_feature = torch.cat(obj_img_feature)
        subj_img_feature = subj_img_feature.view(subj_img_feature.size(0), -1)
        obj_img_feature = obj_img_feature.view(obj_img_feature.size(0), -1)
        subj_img_feature = F.relu(self.batchnorm1(self.linear1(subj_img_feature)))
        obj_img_feature = F.relu(self.batchnorm2(self.linear2(obj_img_feature)))

        x_s = torch.cat(
            [
                classeme_subj * self.scale_factor[0],
                t_s * self.scale_factor[1],
                subj_img_feature * self.scale_factor[2],
            ],
            1,
        )
        x_o = torch.cat(
            [
                classeme_obj * self.scale_factor[0],
                t_o * self.scale_factor[1],
                obj_img_feature * self.scale_factor[2],
            ],
            1,
        )

        v_s = F.relu(self.W_s(x_s))
        v_o = F.relu(self.W_o(x_o))

        res = self.W_p(v_o - v_s)
        return res

        # return torch.sum(self.W_p(v_o - v_s) * predicates, 1)

    