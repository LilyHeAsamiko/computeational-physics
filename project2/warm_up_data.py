# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:39:31 2019

@author: user
"""

import h5py
import numpy as np
f1 = h5py.File('warm_up_data.h5','r+');
print('keys: %s' % list(f1.keys()))
for d in f1["data"]:
    print (d)
warm_up = open("warm_up.txt","w+")
for data in f1["data"]:
    warm_up.write("%s\n" % repr(data));
for x_grid in f1["x_grid"]:
    warm_up.write("%s\n\n" % repr(x_grid));
for y_grid in f1["y_grid"]:
    warm_up.write("%s\n" % repr(y_grid));

warm_up.close()
