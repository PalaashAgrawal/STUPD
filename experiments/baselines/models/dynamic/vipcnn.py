import math
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
from torchvision.ops import roi_pool as torch_roipool

import pdb




class VipCNN_utils(nn.Module):
    def __init__(self):
        super().__init__()

    def roi_pool(self, x, rois, out_size):
        f'rois = bbox coordinates [hmin, hmax, wmin, wmax]'
        
        H,W = x.shape[-2], x.shape[-1]

        
        #torch_roipool expects bboxes in wmin, hmin, wmax, hmax format. So we rearrange the columns in the tensor using this function
        rois = torch.index_select(rois, 1, torch.LongTensor([2,0,3,1]).to(self.device)) 

        rois = rois*torch.Tensor([W,H,W,H]).to(self.device)
        res = torch_roipool(x, [roi.view(1,-1) for roi in rois], out_size)
        return res
      

    def _union_bbox(self, bbox_a, bbox_b):
        return torch.cat(
            [
            torch.min(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
            torch.max(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
            torch.min(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
            torch.max(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1),
            ],
            dim=1)


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

        return bbox
        
    # def intersection_bbox(self, bbox_a, bbox_b):
    #     return torch.cat(
    #         [
    #             torch.max(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
    #             torch.min(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
    #             torch.max(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
    #             torch.min(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1),
    #         ],
    #         dim=1,
    #     )



class VipCNN(VipCNN_utils):
    def __init__(self, roi_size, num_classes, pretrained = True):
        f'''
        PAg: In our version, we keep all backbones as resnet18. 

        

        Default values:
        roi_size = 6

        pretrained: whether or not image feature extraction backbone is pretrained. If True, Kinetics400_V1 weights are used for the 3d module

        '''
        super().__init__()

        self.c = num_classes
        self.model = 'vipcnn'
        self.roi_size = roi_size

        # weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        # model = models.__dict__['resnet18'](weights=weights)

        weights = models.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = models.__dict__['r3d_18'](weights=weights)



        children = list(model.children())
        # self.shared_conv_layers = nn.Sequential(*children[:7])
        self.shared_conv_layers = nn.Sequential(*children[:4])

        # self.pre_pmps1_so = children[7]
        self.pre_pmps1_so = children[4]
        self.pre_pmps1_p = copy.deepcopy(self.pre_pmps1_so)





        self.pmps1_conv_so = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.pmps1_conv_p = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)


        self.pmps1_gather_batchnorm_so = nn.BatchNorm2d(128)
        self.pmps1_gather_batchnorm_p = nn.BatchNorm2d(128)

        self.pmps1_conv_so2p = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pmps1_conv_p2s = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pmps1_conv_p2o = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.pmps1_broadcast_batchnorm_p = nn.BatchNorm2d(128)
        self.pmps1_broadcast_batchnorm_s = nn.BatchNorm2d(128)
        self.pmps1_broadcast_batchnorm_o = nn.BatchNorm2d(128)

        self.pmps2_gather_linear_so = nn.Linear(256 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_gather_linear_p = nn.Linear(256 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_linear_s2p = nn.Linear(32 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_linear_o2p = nn.Linear(32 * roi_size * roi_size, 32 * roi_size * roi_size)
        
        # self.pmps2_broadcast_linear_so = nn.Linear(64 * roi_size * roi_size, 8 * roi_size * roi_size)
        self.pmps2_broadcast_linear_p = nn.Linear(32 * roi_size * roi_size, 4 * roi_size * roi_size)
        
        self.pmps2_gather_batchnorm_s = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_gather_batchnorm_o = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_gather_batchnorm_p = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_broadcast_batchnorm_p = nn.BatchNorm1d(4 * roi_size * roi_size)

        self.fc = nn.Linear(4 * roi_size * roi_size, self.c)

        self.device = None


    def forward(self, img, bbox_s, bbox_o):

        if self.device is None: self.device = next(self.parameters()).device

        shared_feature_maps = self.shared_conv_layers(img) 


        pre_pmps1_feature_so = self.pre_pmps1_so(shared_feature_maps)
        pre_pmps1_feature_p = self.pre_pmps1_p(shared_feature_maps)


        #PAg: reshaping the feature maps from 3d to 2d
        pre_pmps1_feature_so = pre_pmps1_feature_so.view(pre_pmps1_feature_so.shape[0], -1, pre_pmps1_feature_so.shape[-2], pre_pmps1_feature_so.shape[-1])
        pre_pmps1_feature_p = pre_pmps1_feature_p.view(pre_pmps1_feature_p.shape[0], -1, pre_pmps1_feature_p.shape[-2] ,pre_pmps1_feature_p.shape[-1])

        # gather
        pmps1_feature_so = self.pmps1_conv_so(pre_pmps1_feature_so) #output = Nx128xAxA 
        pmps1_gather_so = F.relu(self.pmps1_gather_batchnorm_so(pmps1_feature_so)) # output = Nx128xAxA


        pmps1_feature_p = self.pmps1_conv_p(pre_pmps1_feature_p) #Nx128xAxA
        pmps1_gather_p = F.relu(self.pmps1_gather_batchnorm_p(
                                                            pmps1_feature_p + self.pmps1_conv_so2p(pmps1_gather_so))
                                                            )

        # broadcast
        pmps1_broadcast_p = F.relu(self.pmps1_broadcast_batchnorm_p(pmps1_feature_p))
        pmps1_broadcast_s = F.relu(
            self.pmps1_broadcast_batchnorm_s(
                pmps1_feature_so + self.pmps1_conv_p2s(pmps1_broadcast_p)
            )
        )

        pmps1_broadcast_o = F.relu(
            self.pmps1_broadcast_batchnorm_o(
                pmps1_feature_so + self.pmps1_conv_p2o(pmps1_broadcast_p)
            )
        )

        # concat
        post_pmps1_feature_s = torch.cat([pmps1_gather_so, pmps1_broadcast_s], dim=1) #Nx256xAxA
        post_pmps1_feature_o = torch.cat([pmps1_gather_so, pmps1_broadcast_o], dim=1)
        post_pmps1_feature_p = torch.cat([pmps1_gather_p, pmps1_broadcast_p], dim=1)


        # PAg: just using the first bounding box from the list of bboxess
        bbox_s, bbox_o = bbox_s[:,0,:], bbox_o[:,0,:]
        bbox_s, bbox_o = bbox_s.view(-1, bbox_s.shape[-1]), bbox_o.view(-1, bbox_o.shape[-1])

        #PAg: addded this line which was not there in the original SpatialSense implementation. 
        bbox_s, bbox_o = self._normalize_bbox(bbox_s, img.shape), self._normalize_bbox(bbox_o, img.shape)




        

        # RoI pooling
        post_pool_feature_s = self.roi_pool(post_pmps1_feature_s, bbox_s, self.roi_size)
        post_pool_feature_s = post_pool_feature_s.view(post_pool_feature_s.size(0), -1)

        post_pool_feature_o = self.roi_pool(post_pmps1_feature_o, bbox_o, self.roi_size)
        post_pool_feature_o = post_pool_feature_o.view(post_pool_feature_o.size(0), -1)

        post_pool_feature_p = self.roi_pool(post_pmps1_feature_p, self._union_bbox(bbox_s, bbox_o), self.roi_size)
        post_pool_feature_p = post_pool_feature_p.view(post_pool_feature_p.size(0), -1)

        # gather
        pmps2_gather_s = F.relu(
            self.pmps2_gather_batchnorm_s(
                self.pmps2_gather_linear_so(post_pool_feature_s)
            )
        )
        pmps2_gather_o = F.relu(
            self.pmps2_gather_batchnorm_o(
                self.pmps2_gather_linear_so(post_pool_feature_o)
            )
        )
        pmps2_gather_p = F.relu(
            self.pmps2_gather_batchnorm_p(
                self.pmps2_gather_linear_p(post_pool_feature_p)
                + self.pmps2_linear_s2p(pmps2_gather_s)
                + self.pmps2_linear_o2p(pmps2_gather_o)
            )
        )

        # broadcast
        pmps2_broadcast_p = F.relu(
            self.pmps2_broadcast_batchnorm_p(
                self.pmps2_broadcast_linear_p(pmps2_gather_p)
            )
        )
        # pmps2_broadcast_s = F.relu(self.pmps2_broadcast_linear_so(pmps2_gather_s) + self.pmps2_linear_p2s(pmps2_broadcast_p))
        # pmps2_broadcast_o = F.relu(self.pmps2_broadcast_linear_so(pmps2_gather_o) + self.pmps2_linear_p2o(pmps2_broadcast_p))

        logits = self.fc(pmps2_broadcast_p)
        return logits
        # return torch.sum(logits * predicate, 1)
