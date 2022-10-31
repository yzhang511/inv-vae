import torch
from torch import nn
from torch.nn import functional as F

from inv_vae.utils.helpers import to_var


class GraphConv(nn.Linear):
    '''
    graph convolutional layers.
    ----
    inputs:
    
    ----
    outputs:
    
    '''
    def __init__(self, in_feat_dim, out_feat_dim, device, bias=True):
        super(GraphConv, self).__init__(in_feat_dim, out_feat_dim, device)
        self.mask_flag = False
        self.device = device
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data.to(self.device) * self.mask.data
        self.mask_flag = True 
        
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight.to(self.device) * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)