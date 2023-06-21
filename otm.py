# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as tF
from simple_ot import SampleOT

eps = 1e-12
class L2_DIS:
    factor = 1 / 32
    @staticmethod
    def __call__(X, Y):
        '''
        X.shape = (batch, M, D)
        Y.shape = (batch, N, D)
        returned cost matrix's shape is ()
        '''
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = ((x_col - y_row) ** 2).sum(dim=-1) / 2
        return C * L2_DIS.factor

    @staticmethod
    def barycenter(weight, coord):
        '''
        weight.shape = (batch, M, N)
        coord.shape = (batch, M, D)
        returned coord's shape is (batch, N D)
        '''
        weight = weight / (weight.sum(dim=1, keepdim=True) + eps)
        return weight.permute(0, 2, 1) @ coord

blur = 0.01
per_cost = L2_DIS()
ot = SampleOT(blur=blur, scaling=0.9, reach=None, fixed_epsilon=False)

def den2coord(denmap, scale_factor=8):
    coord = torch.nonzero(denmap > eps)
    denval = denmap[coord[:, 0], coord[:, 1]]
    if scale_factor != 1:
        coord = coord.float() * scale_factor + scale_factor / 2
    return denval.reshape(1, -1, 1), coord.reshape(1, -1, 2)

def init_dot(denmap, n, scale_factor=8):

    norm_den = denmap[None, None, ...]
    norm_den = tF.interpolate(norm_den, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    norm_den = norm_den[0, 0]

    d_coord = torch.nonzero(norm_den > eps)
    norm_den = norm_den[d_coord[:, 0], d_coord[:, 1]]

    cidx = torch.multinomial(norm_den, num_samples=n, replacement=False)
    coord = d_coord[cidx]
    
    B = torch.ones(1, n, 1).to(denmap)
    B_coord = coord.reshape(1, n, 2)
    return B, B_coord

@torch.no_grad()
def OT_M(A, A_coord, B, B_coord, scale_factor=8, max_itern=8):
    for iter in range(max_itern):
        # OT-step
        C = per_cost(A_coord, B_coord)
        F, G = ot(A, B, C)
        PI = ot.plan(A, B, F, G, C)
        # M-step
        nB_coord = per_cost.barycenter(PI, A_coord)
        move = torch.norm(nB_coord - B_coord, p=2, dim=-1)
        if move.mean().item() < 1 and move.max().item() < scale_factor:
            break
        B_coord = nB_coord
    
    return (nB_coord).reshape(-1, 2)

@torch.no_grad()
def den2seq(denmap, scale_factor=8, max_itern=16, ot_scaling=0.75):
    ot.scaling = ot_scaling
    assert denmap.dim() == 2, f"the shape of density map should be [H, W], but the given one is {denmap.shape}"
    
    num = int(denmap.sum().item() + 0.5)
    if num < 0.5:
        return torch.zeros((0, 2)).to(denmap)

    # normalize density map
    denmap = denmap * num / denmap.sum()
    
    A, A_coord = den2coord(denmap, scale_factor)
    B, B_coord = init_dot(denmap, num, scale_factor)

    flocs = OT_M(A, A_coord, B, B_coord, scale_factor, max_itern=max_itern)
    return flocs

@torch.no_grad()
def main():
    import cv2
    import os
    import matplotlib.pyplot as plt
    datadir = 'samples'
    imlist = [14, 77]
    for idx in imlist:
        img = cv2.imread(os.path.join(datadir, f'IMG_{idx}.jpg'))
        imh, imw = img.shape[:2]
        denmap = torch.load(os.path.join(datadir, f"{idx}.pth"))
        dh, dw = denmap.shape
        scale_factor = imw / dw
        print(img.shape, denmap.shape, scale_factor)
        plt.imsave(f"denmap{idx}.png", denmap.cpu(), cmap='jet')
        dot = den2seq(denmap, scale_factor)
        
        # the output's axis is (h, w)
        dot_coord = dot.long().cpu()
        dotmap = torch.zeros((imh, imw))
        dotmap[dot_coord[:, 0], dot_coord[:, 1]] = 1
        dotmap = tF.conv2d(dotmap[None, None, ...], torch.ones((1, 1, 5, 5)), padding=2)[0, 0]
        
        
        plt.imsave(f"dotmap{idx}.png", dotmap)
    
if __name__ == '__main__':
    main()