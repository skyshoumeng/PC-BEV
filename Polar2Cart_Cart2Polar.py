import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 


### Pytorch implementation for Polar2Cart and Cart2Polar operation.
### We reimplement these operations in triton for inference and speed testing.

class Polar2Cart(nn.Module):
    def __init__(self, polar_size, cart_size, center_drop_grid_size=3., max_batch=4, mode='in'):
        super(Polar2Cart, self).__init__()
        if mode=='in':
            ## gird in polar
            yy_org, xx_org = torch.meshgrid(torch.arange(cart_size[0]), torch.arange(cart_size[1]))
            yy = yy_org - cart_size[0] / 2. + 0.5
            xx = xx_org - cart_size[1] / 2. + 0.5

            depth = torch.sqrt(xx ** 2 + yy ** 2)
            phi = np.pi - torch.arctan2(yy, xx)
            index_y = depth / (cart_size[0] / 2. * np.sqrt(2.)) * (polar_size[0] + center_drop_grid_size) - center_drop_grid_size
            index_x = phi / np.pi / 2. * polar_size[1]

            mask = (index_y > 0).view(-1)
            index_y = index_y / polar_size[0] * 2. - 1.
            index_x = index_x / polar_size[1] * 2. - 1.
            grid_sample_index = torch.stack((index_x, -index_y), axis=-1).view(-1,2) 
            grid_sample_index = grid_sample_index[mask, :].unsqueeze(0).unsqueeze(2)
            self.grid_sample_index = nn.Parameter(grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False)

            grid_sample_xy = torch.stack((yy_org, xx_org), axis=-1).view(-1,2)
            grid_sample_xy = grid_sample_xy[mask, :]
            batch_index = torch.arange(max_batch).unsqueeze(1).repeat(1,grid_sample_xy.shape[0]).view(-1,1)
            self.grid_sample_xy = nn.Parameter(torch.cat([batch_index, grid_sample_xy.unsqueeze(0).repeat(max_batch,1,1).view(-1,2)], dim=1).long(), requires_grad=False)

            self.length_per = grid_sample_xy.shape[0]
        elif mode=='out':
            ## gird out polar
            yy_org, xx_org = torch.meshgrid(torch.arange(cart_size[0]), torch.arange(cart_size[1]))
            yy = yy_org - cart_size[0] / 2. + .5
            xx = xx_org - cart_size[1] / 2. + .5

            depth = torch.sqrt(xx ** 2 + yy ** 2)
            phi = np.pi - torch.arctan2(yy, xx)
            index_y = depth / (cart_size[0] / 2. ) * (polar_size[0] + center_drop_grid_size) - center_drop_grid_size
            index_x = phi / np.pi / 2. * polar_size[1]

            mask = (index_y > 0.0).view(-1) & (index_y < polar_size[0]-1.).view(-1)
            index_y = index_y / polar_size[0] * 2. - 1.
            index_x = index_x / polar_size[1] * 2. - 1.

            grid_sample_index = torch.stack((index_x, -index_y), axis=-1).view(-1,2)
            grid_sample_index = grid_sample_index[mask, :].unsqueeze(0).unsqueeze(2)
            self.grid_sample_index = nn.Parameter(grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False)

            grid_sample_xy = torch.stack((yy_org, xx_org), axis=-1).view(-1,2)
            grid_sample_xy = grid_sample_xy[mask, :]
            batch_index = torch.arange(max_batch).unsqueeze(1).repeat(1,grid_sample_xy.shape[0]).view(-1,1)
            self.grid_sample_xy = nn.Parameter(torch.cat([batch_index, grid_sample_xy.unsqueeze(0).repeat(max_batch,1,1).view(-1,2)], dim=1).long(), requires_grad=False)
    
            self.length_per = grid_sample_xy.shape[0]
    def forward(self, polar_feat, ref_feat):
        
        grid_feat = ref_feat.clone() * 1.
        length = self.length_per * ref_feat.shape[0]
        grid_feat[self.grid_sample_xy[:length,0], :, self.grid_sample_xy[:length,1], self.grid_sample_xy[:length,2]] = \
                    F.grid_sample(polar_feat, self.grid_sample_index[:ref_feat.shape[0],...], mode='bilinear', padding_mode='zeros', align_corners=True).permute(0,2,1,3).reshape(-1, ref_feat.shape[1])

        return grid_feat
    

class Cart2Polar(nn.Module):
    def __init__(self, polar_size, cart_size, center_drop_grid_size=3., max_batch=4, mode='in'):
        super(Cart2Polar, self).__init__()
        if mode=='in':
            ## polar in grid
            yy, xx = torch.meshgrid(torch.arange(polar_size[0]), torch.arange(polar_size[1]))
            theta = np.pi - xx / polar_size[1] * np.pi * 2.

            index_x = (polar_size[0]-0.5-yy+center_drop_grid_size) * torch.cos(theta) / (polar_size[0]+center_drop_grid_size) * cart_size[0] / 2. + cart_size[0] / 2. #- 0.5
            index_y = (polar_size[0]-0.5-yy+center_drop_grid_size) * torch.sin(theta) / (polar_size[0]+center_drop_grid_size) * cart_size[0] / 2. + cart_size[0] / 2. #- 0.5

            grid_sample_index = torch.stack((index_x, index_y), axis=-1).view(-1,2).unsqueeze(0).unsqueeze(2) / cart_size[0] * 2. - 1.
            self.grid_sample_index = nn.Parameter(grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False)

            grid_sample_xy = torch.stack((yy, xx), axis=-1).view(-1,2)
            batch_index = torch.arange(max_batch).unsqueeze(1).repeat(1,grid_sample_xy.shape[0]).view(-1,1)
            self.grid_sample_xy = nn.Parameter(torch.cat([batch_index, grid_sample_xy.unsqueeze(0).repeat(max_batch,1,1).view(-1,2)], dim=1).long(), requires_grad=False)

            self.length_per = grid_sample_xy.shape[0]
        elif mode=='out':
            yy, xx = torch.meshgrid(torch.arange(polar_size[0]), torch.arange(polar_size[1]))
            theta = np.pi - xx / polar_size[1] * np.pi * 2.

            index_x = (polar_size[0]-0.5-yy+center_drop_grid_size) * torch.cos(theta) / (polar_size[0]+center_drop_grid_size) * np.sqrt(2) * cart_size[0] / 2. + cart_size[0] / 2. #- 0.5
            index_y = (polar_size[0]-0.5-yy+center_drop_grid_size) * torch.sin(theta) / (polar_size[0]+center_drop_grid_size) * np.sqrt(2) * cart_size[0] / 2. + cart_size[0] / 2. #- 0.5

            grid_sample_index = torch.stack((index_x, index_y), axis=-1).view(-1,2) / cart_size[0] * 2. - 1.
            mask = ((index_y>0)&(index_x>0)&(index_y<cart_size[0])&(index_x<cart_size[0])).view(-1)
            grid_sample_index = grid_sample_index[mask, :].unsqueeze(0).unsqueeze(2)

            self.grid_sample_index = nn.Parameter(grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False)

            grid_sample_xy = torch.stack((yy, xx), axis=-1).view(-1,2)
            grid_sample_xy = grid_sample_xy[mask, :]
            batch_index = torch.arange(max_batch).unsqueeze(1).repeat(1,grid_sample_xy.shape[0]).view(-1,1)
            self.grid_sample_xy = nn.Parameter(torch.cat([batch_index, grid_sample_xy.unsqueeze(0).repeat(max_batch,1,1).view(-1,2)], dim=1).long(), requires_grad=False)

            self.length_per = grid_sample_xy.shape[0]

    def forward(self, grid_feat, ref_feat):

        polar_feat = ref_feat.clone() * 1.
        length = self.length_per * ref_feat.shape[0]
        polar_feat[self.grid_sample_xy[:length,0], :, self.grid_sample_xy[:length,1], self.grid_sample_xy[:length,2]] = \
                    F.grid_sample(grid_feat, self.grid_sample_index[:ref_feat.shape[0],...], mode='bilinear', padding_mode='zeros', align_corners=True).permute(0,2,1,3).reshape(-1, ref_feat.shape[1])
        return polar_feat
