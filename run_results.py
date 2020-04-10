
import numpy as np
import os
import copy
import threading
import argparse

from results import get_runs

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--print', type=int, default=0)
cmd_args = parser.parse_args()

##############################################

num_gpus = 5
counter = 1

def run_command(param):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    name = '%s_%d_%f_%f' % (
            param['benchmark'],
            param['batch_size'],
            param['lr'],
            param['noise']
            )
             
    cmd = "python3 %s --gpu %d --epochs %d --batch_size %d --lr %f --eps %f --noise %f --name %s" % (
           param['benchmark'], 
           gpu, 
           param['epochs'], 
           param['batch_size'], 
           param['lr'], 
           param['eps'], 
           param['noise'], 
           name
           )

    if cmd_args.print:
        print (cmd)
    else:
        os.system(cmd)

    return

##############################################

runs = get_runs()

##############################################

num_runs = len(runs)
parallel_runs = num_gpus

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=(args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
