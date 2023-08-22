import math
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
# from .vipcnn import union_bbox, intersection_bbox
import pdb


#Had to do some hacks in this script. all F.avg_pool2d in the roi pool functions contain the entire height and width of input feature map (instead of just bbox)


class PPRFCN_utils(nn.Module):
    def __init__(self):
        super().__init__()

    def _union_bbox(self, bbox_a, bbox_b):
        return torch.cat(
            [
                torch.min(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
                torch.max(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
                torch.min(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
                torch.max(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1),
            ],
            dim=1,
        )

    def _intersection_bbox(self, bbox_a, bbox_b):
        return torch.cat(
            [
                torch.max(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
                torch.min(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
                torch.max(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
                torch.min(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1),
            ],
            dim=1,
        )





    def _subj_obj_pool(self, prs_maps, bbox):
        batchsize, C, H, W = prs_maps.shape
        feature = torch.Tensor(batchsize, 9, 3, 3).to(prs_maps.device)

        #PAg: made this logic shorter. Just use a 3d avgpool
        pool = F.avg_pool3d( (prs_maps[None]).squeeze(dim=0), #avgpool2d expects an input of N,B,Ci,Hi,Wi
                                kernel_size = (9 ,int(H/3.0), int(W/3.0)),
                                )
        return pool
        


        # for n in range(batchsize):
        #     cell_height = (bbox[n, 1] - bbox[n, 0]) / 3.0
        #     cell_width = (bbox[n, 3] - bbox[n, 2]) / 3.0
        #     for i in range(3):
        #         for j in range(3):

        #             bbox_cell = [
        #                 int(prs_maps.size(2) * (bbox[n, 0] + i * cell_height).item()),
        #                 int(prs_maps.size(2) * (bbox[n, 0] + (i + 1) * cell_height).item()),
        #                 int(prs_maps.size(3) * (bbox[n, 2] + j * cell_width).item()),
        #                 int(prs_maps.size(3) * (bbox[n, 2] + (j + 1) * cell_width).item()),
        #             ]

        #             if bbox_cell[1] <= bbox_cell[0]:
        #                 if bbox_cell[1] < prs_maps.size(2) - 1: bbox_cell[1] += 1
        #                 else: bbox_cell[0] -= 1

        #             if bbox_cell[3] <= bbox_cell[2]:
        #                 if bbox_cell[3] < prs_maps.size(3): bbox_cell[3] += 1
        #                 else: bbox_cell[2] -= 1

                    
        #             #PAg
        #             if bbox_cell[1]==bbox_cell[0]: bbox_cell[1]+=1
        #             if bbox_cell[3]==bbox_cell[2]: bbox_cell[3]+=1


        #             start = i * 27 + j * 9
        #             cell_input = prs_maps[
        #                 n,
        #                 start : start + 9,
        #                 # bbox_cell[0] : bbox_cell[1],
        #                 :,
        #                 # bbox_cell[2] : bbox_cell[3],
        #                 :,
        #             ]

        #             feature[n, :, i, j] = F.avg_pool2d(
        #                 cell_input, (cell_input.size(1), cell_input.size(2)), 
        #             ).squeeze()

        # return feature

    
    def _joint_pool(self, prs_maps_joint_s, prs_maps_joint_o, bbox_s, bbox_o):
        # bbox_union = self._union_bbox(bbox_s, bbox_o)
        batchsize, C, H, W = prs_maps_joint_s.shape
        # batchsize = prs_maps_joint_s.size(0)

        nc =  prs_maps_joint_s.size(1)

        # feature = torch.zeros(batchsize, 9, 3, 3).to(prs_maps_joint_s.device)
        # feature = torch.zeros(batchsize, nc, 3, 3).to(prs_maps_joint_s.device)

        #PAg: made this logic shorter. Just use a 3d avgpool
        pool_s = F.avg_pool3d(    (prs_maps_joint_s[None]).squeeze(dim=0), #avgpool2d expects an input of N,B,Ci,Hi,Wi
                                kernel_size = (9 ,int(H/3.0), int(W/3.0)),
                                )
                            
        pool_o = F.avg_pool3d(   (prs_maps_joint_o[None]).squeeze(dim=0), #avgpool2d expects an input of N,B,Ci,Hi,Wi
                                kernel_size = (9 ,int(H/3.0), int(W/3.0)),
                                )

        feature = pool_s + pool_o
            
        return feature

        

        


        # for n in range(batchsize):
        #     cell_height = (bbox_union[n, 1] - bbox_union[n, 0]) / 3.0
        #     cell_width = (bbox_union[n, 3] - bbox_union[n, 2]) / 3.0
        #     for i in range(3):
        #         for j in range(3):
        #             start = i * 27 + j * 9
        #             bbox_cell = [
        #                 (bbox_union[n, 0] + i * cell_height).item(),
        #                 (bbox_union[n, 0] + (i + 1) * cell_height).item(),
        #                 (bbox_union[n, 2] + j * cell_width).item(),
        #                 (bbox_union[n, 2] + (j + 1) * cell_width).item(),
        #             ]
        #             # subject
        #             I_s = self._intersection_bbox(
        #                 bbox_s[n].unsqueeze(0),
        #                 torch.Tensor(bbox_cell).to(bbox_s.device).unsqueeze(0),
        #             ).squeeze()
        #             I_s = [
        #                 int(prs_maps_joint_s.size(2) * I_s[0].item()),
        #                 int(prs_maps_joint_s.size(2) * I_s[1].item()),
        #                 int(prs_maps_joint_s.size(3) * I_s[2].item()),
        #                 int(prs_maps_joint_s.size(3) * I_s[3].item()),
        #             ]
        #             if I_s[0] < I_s[1] and I_s[2] < I_s[3]:
        #                 cell_input = prs_maps_joint_s[
        #                     n, start : start + 9, 
        #                     # I_s[0] : I_s[1], I_s[2] : I_s[3]
        #                     :,:
        #                 ]
        #                 feature[n, :, i, j] += F.avg_pool2d(
        #                     cell_input, (cell_input.size(1), cell_input.size(2))
        #                 ).squeeze()

                    
        #             # object
        #             I_o = self._intersection_bbox(
        #                 bbox_o[n].unsqueeze(0),
        #                 torch.Tensor(bbox_cell).to(bbox_o.device).unsqueeze(0),
        #             ).squeeze()
        #             I_o = [
        #                 int(prs_maps_joint_o.size(2) * I_o[0].item()),
        #                 int(prs_maps_joint_o.size(2) * I_o[1].item()),
        #                 int(prs_maps_joint_o.size(3) * I_o[2].item()),
        #                 int(prs_maps_joint_o.size(3) * I_o[3].item()),
        #             ]


        #             if I_o[0] < I_o[1] and I_o[2] < I_o[3]:
        #                 cell_input = prs_maps_joint_o[
        #                     n, start : start + 9, 
        #                     # I_o[0] : I_o[1], I_o[2] : I_o[3]
        #                     :,:,
        #                 ]
        #                 feature[n, :, i, j] += F.avg_pool2d(
        #                     cell_input, (cell_input.size(1), cell_input.size(2))
        #                 ).squeeze()



        # return feature



    
    def _normalize_bbox(self, bbox, img_shape):
        f'''
        PAg: ROI pooling functions (such as self._subj_obj_pool and self._joint_pool scale bboxes to the shape of output matrix by multiplying. 
        If we don't divide by the original image size first, 
        then the multiplication will lead to extremely large values, greater than the size of the output matrix. 
        This will throw error. 

        Hence this function
        '''
        H,W = img_shape[-2], img_shape[-1]
        bbox=torch.div(bbox,torch.Tensor([H,H,W,W]).to(self.device))
        return bbox





class convert_3d_to_2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        N,_,_,H,W = x.shape
        return x.view(N,-1, H, W)

class PPRFCN(PPRFCN_utils):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()

        self.c = num_classes
        weights = models.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = models.__dict__['r3d_18'](weights=weights)
        self.backbone = nn.Sequential(
                *list(model.children())[:-2],

                convert_3d_to_2d(),

                nn.Conv2d(512, 256, kernel_size=1, stride=1)
            )

        # if backbone == "resnet18":
        #     self.resnet = nn.Sequential(
        #         *list(models.resnet18(pretrained=True).children())[:-2],
        #         nn.Conv2d(512, 256, kernel_size=1, stride=1)
        #     )
        # elif backbone == "resnet101":
        #     self.resnet = nn.Sequential(
        #         *list(models.resnet101(pretrained=True).children())[:-2],
        #         nn.Conv2d(2048, 256, kernel_size=1, stride=1)
        #     )

        self.conv = nn.Conv2d(
            256, 324, kernel_size=1, stride=1
        )  # subject : object : joint_s : joint_o 4 k^2 R
        self.batchnorm = nn.BatchNorm2d(324)
        self.fc = nn.Sequential(nn.Linear(81, 40), nn.BatchNorm1d(40), nn.Linear(40, self.c))
        self.device = None
    

    def forward(self, img, bbox_s, bbox_o):

        if self.device is None: self.device = next(self.parameters()).device

        shared_feature_maps = self.backbone(img)  # batchsize x 256 x 23 x 23
        prs_maps = self.batchnorm(
            self.conv(shared_feature_maps)
        )  # batchsize x 324 x 23 x 23

        prs_maps_s = prs_maps[:, :81, :, :]
        prs_maps_o = prs_maps[:, 81:162, :, :]
        prs_maps_joint_s = prs_maps[:, 162:243, :, :]
        prs_maps_joint_o = prs_maps[:, 243:, :, :]


        # PAg: just using the first bounding box from the list of bboxess
        bbox_s, bbox_o = bbox_s[:,0,:], bbox_o[:,0,:]
        bbox_s, bbox_o = bbox_s.view(-1, bbox_s.shape[-1]), bbox_o.view(-1, bbox_o.shape[-1])

        bbox_s, bbox_o = self._normalize_bbox(bbox_s, img.shape), self._normalize_bbox(bbox_o, img.shape)
        

        
        # subject/object pooling
        feature_s = self._subj_obj_pool(prs_maps_s, bbox_s)  # 3 x 3 x R
        feature_o = self._subj_obj_pool(prs_maps_o, bbox_o)  # 3 x 3 x R

        # joint pooling
        feature_joint = self._joint_pool(prs_maps_joint_s, prs_maps_joint_o, bbox_s, bbox_o)

        # vote
        # logits = (F.avg_pool2d(feature_s, 3) + F.avg_pool2d(feature_o, 3) + F.avg_pool2d(feature_joint, 3)).squeeze()
        #

        logits = self.fc(
            (feature_s + feature_o + feature_joint).view(feature_s.size(0), -1)
        )

        return logits 


    def __name__(self): return f'PPR-FCN Model'