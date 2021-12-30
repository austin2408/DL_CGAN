import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import sys
import math
import random
import numpy as np
import time
from torch.nn import init
import os
            
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = 300
        self.ngf = 64
        self.fc = nn.Linear(24,100)
        self.act = nn.ReLU()
        self.network = nn.Sequential(
                nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0), # 
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(self.ngf * 2),
                nn.ReLU(True),

                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),

                # state size. 3 x 64 x 64
                nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self, x, cond):
        cond = cond.permute(0,3,2,1)
        cond = self.act(self.fc(cond.float()).permute(0,3,1,2))
        x = torch.cat([x, cond], 1)
        out = self.network(x.float())

        return out

    def weight_init(self,mean,std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ndf = 64
        self.fc = nn.Linear(24,self.ndf*self.ndf)
        self.act = nn.LeakyReLU()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, self.ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x, cond=None):
        cond = cond.permute(0,2,3,1)
        cond = self.act(self.fc(cond).view(cond.shape[0],1,self.ndf,self.ndf))
        x = torch.cat([x, cond], 1)
        output = self.main(x)

        return output.view(-1, 1).squeeze(0)

    def weight_init(self,mean,std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.ndf = 64
        self.fc = nn.Linear(24,3*self.ndf*self.ndf)
        self.act = nn.LeakyReLU()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(6, self.ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, cond=None):
        cond = cond.permute(0,2,3,1)
        cond = self.act(self.fc(cond).view(cond.shape[0],3,self.ndf,self.ndf))
        x = torch.cat([x, cond], 1)
        output = self.main(x)

        return output.view(-1, 1).squeeze(0)

    def weight_init(self,mean,std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()