# -*- coding: utf-8 -*-

import torch
import numpy as np

EPS =  1e-12

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=1)[0], y.min(dim=1)[0]), dim=1).min(dim=1)[0] # B 2
    maxs = torch.stack((x.max(dim=1)[0], y.max(dim=1)[0]), dim=1).max(dim=1)[0] # B 2
    diameter = (maxs-mins).norm(dim=1).max().item()
    if diameter == 0:
        diameter = 16
    return diameter

def epsilon_schedule(diameter, blur, scaling, fixed_epsilon=False):
    # print("[EPS]:", np.log(diameter), np.log(blur), np.log(scaling))
    schedule = np.arange(np.log(diameter), np.log(blur), np.log(scaling))
    if fixed_epsilon:
        epsilon_s = [ blur ] + [ blur for _ in  schedule] + [ blur ]
    else:
        epsilon_s = [ diameter ] + [ np.exp(e) for e in schedule ] + [ blur ]
    return epsilon_s

def dampening(epsilon, reach):
    return 1 if reach is None else 1 / ( 1 + epsilon / reach )

def softmin(logB, G, C, epsilon):
    B = C.shape[0]
    x = logB.view(B, 1, -1) + (G.view(B, 1, -1) - C) / epsilon
    x = - epsilon * x.logsumexp(2).view(B, -1, 1)
    return x

class SampleOT:
    def __init__(self, blur=0.01, scaling=0.5, reach=None, fixed_epsilon=False) -> None:
        self.blur = blur
        self.scaling = scaling
        self.fixed_epsilon = fixed_epsilon
        self.reach = reach

    @torch.no_grad()        
    def __call__(self, A, B, cost, F=None, G=None, diameter=None):
        '''
        A.shape = B H 1
        B.shape = B W 1
        cost.shape = B H W
        '''
        
        bsize, H, W = cost.shape
        
        fixed_epsilon = (F is not None and G is not None) or self.fixed_epsilon
        diameter = diameter if diameter is not None else cost.max().item()
        diameter = max(8, diameter)
        epsilons = epsilon_schedule(diameter, self.blur, self.scaling, fixed_epsilon)
        
        logA, logB = A.log(), B.log()
        Cab, Cba = cost, cost.permute(0, 2, 1)
        factor = dampening(epsilons[0], self.reach)
        if F is None:
            F = factor * softmin(logB, torch.zeros_like(B), Cab, epsilons[0])
        if G is None:
            G = factor * softmin(logA, torch.zeros_like(A), Cba, epsilons[0])
            
        for i, epsilon in enumerate(epsilons):

            factor = dampening(epsilon, self.reach)
            tF = factor * softmin(logB, G, Cab, epsilon)
            tG = factor * softmin(logA, F, Cba, epsilon)
            F, G = (F + tF) / 2, (G + tG) / 2

        factor = dampening(self.blur, self.reach)
        F, G = factor * softmin(logB, G, Cab, self.blur), factor * softmin(logA, F, Cba, self.blur)
        

        return F.detach(), G.detach()

    def loss(self, A, B, F, G):
        if self.reach is not None:
            F = self.weightfunc(1 - (- F / self.reach).exp())
            G = self.weightfunc(1 - (- G / self.reach).exp())
        return torch.mean( (A * F).flatten(1).sum(dim=1) + (B * G).flatten(1).sum(dim=1) )


    def plan(self, A, B, F, G, cost):
        PI1 = torch.exp((F + G.permute(0, 2, 1) - cost) / self.blur)
        PI2 = A * B.permute(0, 2, 1)
        PI = PI1 * PI2
        return PI