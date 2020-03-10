
import numpy as np
import os
import copy
import threading
import argparse
from results import get_runs

##############################################

runs = get_runs()

##############################################

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    name = '%s_%d_%f_%f' % (
            param['benchmark'], 
            param['batch_size'],
            param['lr'], 
            param['noise']
            )

    res = np.load(name, allow_pickle=True).item()
    key = (param['benchmark'], param['batch_size'], param['lr'], param['noise'])
    val = max(res['acc'])

    print (name, val)
    if key in results.keys():
        if results[key][0] < val:
            results[key] = val
    else:
        results[key] = val
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
