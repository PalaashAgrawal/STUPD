from torch import nn
import torch.nn.functional as F


#_____________________________________________________________________________________________
# ... model description ...


# x = tensor[
#             subj_x - obj_x, subj_y - obj_y, subj_hmin, subj_hmax, subj_wmin, subj_wmax, obj_hmin, obj_hmax, obj_wmin, obj_wmax
#     ]

# x --> linear(10,64) --> linear(64,64)x4  --> predicate_category

#_____________________________________________________________________________________________






def convert_stupd_bbox_to_spatialsense_bbox(bbox):
    f'''
    #stupd bbox format = wmin, hmin, w, h
    #spatialsenses bbox format = hmin, hmax, wmin, wmax

    This is specifically for conversion from stupd format to spatialsense format, which is used for this 2d cordinate model. 
    
    '''

    
    wmin, hmin, w,h = bbox
    return (hmin, hmin+h, wmin, wmin+w)






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
    
    
class coordinateOnlyModel(nn.Module):
    def __init__(self, coord_len, feature_dim, c):
        super().__init__()

        self.layers = nn.Sequential(*[LinearBlock(coord_len, feature_dim)] + [LinearBlock(feature_dim, feature_dim) for _ in range(4)] + [nn.Linear(feature_dim, c)])
    
    def forward(self, x):
        return self.layers(x)