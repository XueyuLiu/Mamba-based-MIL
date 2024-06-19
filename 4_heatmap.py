# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:32:56 2019
Generate heatmaps for test images using the probabilities from the trained model
@author: SCSC
"""

import torch
import sys
import numpy as np
import cv2

# Load probabilities and test data
probs = torch.load('probs_32.pth')
test = torch.load('test-32.lib')
grids = test['grid']
slides = test['slides']
targets = test['targets']

# Reshape the probabilities into heatmaps
temp = []
for i in range(len(probs) // 4624):
    temp.append(probs[i * 4624:(i + 1) * 4624])

heatmaps = []
for i in temp:
    t = []
    for j in range(68):
        t.append(i[68 * j:68 * (j + 1)])
    heatmaps.append(t)
heatmaps = np.array(heatmaps)

# Generate and save heatmaps
for i in range(len(slides)):
    print(f"Processing: [{i + 1}/{len(slides)}] - {slides[i]}")
    sys.stdout.flush()
    
    img = cv2.imread(f'RGB/{slides[i]}')
    heatmap = heatmaps[i]
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_mask = np.uint8(255 * heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.5 + img
    cv2.imwrite(f'heatmap_32/{slides[i]}', superimposed_img)
    cv2.imwrite(f'probmap_32/{slides[i]}', heatmap_mask)
