# -*- coding: utf-8 -*-
"""
Generates train.lib and test.lib
Ensures positive samples are first, negative samples are last
@author: SCSC
"""

import torch
import os
import time
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Record start time
start_time = time.time()

# Define the size of the box for slicing
box_size = 32

# List directories for positive and negative samples
positives = os.listdir('positives')
negatives = os.listdir('negatives')
positives_t = os.listdir('positives_t')
negatives_t = os.listdir('negatives_t')

# Calculate the total number of samples
total_samples = len(positives) + len(negatives) + len(positives_t) + len(negatives_t)
print(total_samples)

# Initialize the dictionaries for train and test libraries
train_library = {}
test_library = {}

def create_sample_list(sample_type, initial_num):
    """
    Creates a list of samples with their grids and targets.

    Args:
        sample_type (str): The type of samples ('positives', 'negatives', etc.).
        initial_num (int): The initial number for tracking progress.

    Returns:
        tuple: Containing grids, slides, targets, and the updated number.
    """
    current_num = initial_num
    slides = []
    grids = []
    targets = []
    file_names = os.listdir(sample_type)

    for file_name in file_names:
        img = Image.open(f'{sample_type}/{file_name}').convert('RGB')
        img = img.resize((1100, 1100)).convert('RGB')
        data = np.array(img)
        height, width = data.shape[:2]

        # Calculate grid coordinates
        h_steps = [(i + 1) * (box_size // 2) for i in range(int((height - box_size) / (box_size // 2)) + 1)]
        if int((height - box_size) / (box_size // 2)) != (height - box_size) / (box_size // 2):
            h_steps.append(height - (box_size // 2))

        w_steps = [(i + 1) * (box_size // 2) for i in range(int((width - box_size) / (box_size // 2)) + 1)]
        if int((width - box_size) / (box_size // 2)) != (width - box_size) / (box_size // 2):
            w_steps.append(width - (box_size // 2))

        hws = [[h, w] for h in h_steps for w in w_steps]
        grids.append(np.array(hws))
        slides.append(file_name)
        targets.append(1 if sample_type in ['positives', 'positives_t'] else 0)
        current_num += 1
        print(current_num / total_samples)

    return grids, slides, targets, current_num

# Create sample lists for train and test datasets
pos_grid, pos_slides, pos_targets, pos_num = create_sample_list('positives', 0)
neg_grid, neg_slides, neg_targets, neg_num = create_sample_list('negatives', pos_num)
pos_gridt, pos_slidest, pos_targetst, post_num = create_sample_list('positives_t', neg_num)
neg_gridt, neg_slidest, neg_targetst, negt_num = create_sample_list('negatives_t', post_num)

# Populate train library dictionary
train_library['targets'] = pos_targets + neg_targets
train_library['slides'] = pos_slides + neg_slides
train_library['grid'] = pos_grid + neg_grid
train_library['size'] = box_size
torch.save(train_library, f'train-{box_size}.lib')

# Populate test library dictionary
test_library['targets'] = pos_targetst + neg_targetst
test_library['slides'] = pos_slidest + neg_slidest
test_library['grid'] = pos_gridt + neg_gridt
test_library['size'] = box_size
torch.save(test_library, f'test-{box_size}.lib')

# Print the elapsed time
print(time.time() - start_time)
