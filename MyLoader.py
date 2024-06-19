# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:59:53 2019

@author: SCSC
"""

import torch
import sys
from PIL import Image
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        size = lib['size'] // 2
        tiles = []
        slideIDX = []
        plen = 0
        self.targets = lib['targets']
        for i,name in enumerate(lib['slides']):
            slide = Image.open('RGB/%s.png'%name)
            slide=slide.resize((1100,1100)).convert('RGB')
            sys.stdout.write('Cutting JPGs: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            grids = lib['grid'][i]
            for grid in grids:
                tiles.append(transform(slide.crop((grid[1]-size,grid[0]-size,grid[1]+size,grid[0]+size))))
            slideIDX.extend([i]*len(grids))
            if self.targets[i] == 1:
                plen += len(grids)
        print('')
        self.tiles = tiles
        self.slideIDX = slideIDX
        self.plen = plen
        print('Number of tiles:%d'%len(slideIDX))
        self.mode = 1
        
    def setmode(self, mode):
        self.mode = mode
    def maketraindata(self, k):
        self.t_data = [(self.slideIDX[x], self.tiles[x], self.targets[self.slideIDX[x]]) for x in k]
    def __getitem__(self,index):
        if self.mode == 1:
            return self.tiles[index], self.targets[self.slideIDX[index]]
        elif self.mode == 2:
            slideIDX, tiles, targets = self.t_data[index]
            return tiles, targets
    def __len__(self):
        if self.mode == 1:
            return len(self.slideIDX)
        elif self.mode == 2:
            return len(self.t_data)

#####################################################################

class OrigDataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        self.targets = lib['targets']
        self.size = lib['size'] // 2
        slides = []
        grid = []
        slideIDX = []
        plen = 0
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening JPGs: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(Image.open('RGB/%s'%name))
            #Flatten grid
            g = lib['grid'][i]
            grid.extend(g)
            slideIDX.extend([i]*len(g))
            if self.targets[i] == 1:
                plen += len(g)
        print('')
        print('Number of tiles: {}'.format(len(grid)))
        self.slides = slides
        self.plen = plen
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].crop((coord[1]-self.size,coord[0]-self.size,coord[1]+self.size,coord[0]+self.size))
            if self.transform is not None:
                img = self.transform(img)
            return img, self.targets[slideIDX]
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].crop((coord[1]-self.size,coord[0]-self.size,coord[1]+self.size,coord[0]+self.size))
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
