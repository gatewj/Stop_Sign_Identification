# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:00:51 2022

@author: Jambo

pip install split-folders
"""

import splitfolders  # or import split_folders

input_folder = 'images/'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
splitfolders.ratio(input_folder,
                   output="images_dir", 
                   seed=42,
                   ratio=(.8, .2), 
                   group_prefix=None) # default values

# If i wanted a validation set
# ratio = (training, validation, test)


# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# enable oversampling of imbalanced datasets, works only with fixed
splitfolders.fixed(input_folder,
                   output="cell_images2", 
                   seed=42,
                   fixed=(35, 20), 
                   oversample=False, group_prefix=None)


# https://www.youtube.com/watch?v=C6wbr1jJvVs
