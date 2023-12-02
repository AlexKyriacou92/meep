import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d
import subprocess
import sys
import configparser

fname_cluster_config = 'config_cluster.txt'
config = configparser.ConfigParser()
config.read(fname_cluster_config)
cluster_settings = config['CLUSTER']

#Cluster Settings
NODES_MIN = int(cluster_settings['NODES_MIN'])
NODES_MAX = int(cluster_settings['NODES_MAX'])
PARTITION = cluster_settings['PARTITION']
DAYS = int(cluster_settings['DAYS'])
HOURS = int(cluster_settings['HOURS'])
MEMORY = int(cluster_settings['MEMORY']) # in MB

NTASKS = -1
NTASKS_PER_NODE = -1
CPUS_PER_TASK = -1

ntasks_str = 'NTASKS'
ntasks_per_node_str = 'NTASKS_PER_NODE'
cpus_per_task_str = 'CPUS_PER_TASK'

if ntasks_str in cluster_settings.keys():
    NTASKS = int(cluster_settings[ntasks_str])
else:
    NTASKS = -1 #OR 1?

if ntasks_per_node_str in cluster_settings.keys():
    NTASKS_PER_NODE = int(cluster_settings[ntasks_per_node_str])
else:
    NTASKS_PER_NODE = -1 #OR 1?

if cpus_per_task_str in cluster_settings.keys():
    CPUS_PER_TASK = int(cluster_settings[cpus_per_task_str])
else:
    CPUS_PER_TASK = -1

def make_job(fname_shell, fname_outfile, jobname, command,
             nNodes_min=NODES_MIN, nNodes_max=NODES_MAX, partition=PARTITION,
             days=DAYS, hours=HOURS, nodeMemory=MEMORY,
             nTasks=NTASKS, nTasks_per_node = NTASKS_PER_NODE, cpus_per_task=CPUS_PER_TASK):
    '''
    Function creates a shell file to run 1 job on Pleaides

    Arguments
    fname_shell : name of shell script <path/to/file.sh>
    fname_output : name of output file (where print statements go) <path/to/file.out>
    jobname : label of job (can be anything but should be consistent with fname_shell and fname_outfile)
    command : python command that runs job

    nNodes_min : minimum number of nodes in Pleaides to run job on
    nNodes_max : maximum number of nodes
    partition : set partition location ('short', 'long' or 'normal')

    --ntasks=# : Number of "tasks" (use with distributed parallelism).

    --ntasks-per-node=# : Number of "tasks" per node (use with distributed parallelism).

    --cpus-per-task=# : Number of CPUs allocated to each task (use with shared memory parallelism).


    days : maximum number of days to run job (kill if it exceeds this + hours)
    hours: (+ hours)
    nodeMemory : how much RAM to assign to node
    '''
    sbatch = "#SBATCH"
    fout = open(fname_shell, 'w+') #create shell file

    #Write parameters of Sbatch
    fout.write("#!/bin/sh\n")

    minutes = 0
    seconds = 0
    fout.write(sbatch + " --job-name=" + jobname + "\n")
    fout.write(sbatch + " --partition=" + partition + "\n")
    fout.write(sbatch + " --time=" + str(days) + "-" + str(hours) + ":" + str(minutes) + ":" + str(
        seconds) + " # days-hours:minutes:seconds\n")
    if nNodes_min == nNodes_max:
        fout.write(sbatch + " --nodes=" + str(nNodes_min) + "\n")
    else:
        fout.write(sbatch + " --nodes=" + str(nNodes_min) + "-" + str(nNodes_max) + "\n")
    fout.write(sbatch + " --mem-per-cpu=" + str(nodeMemory) + " # in MB\n")

    if nTasks != -1:
        fout.write(sbatch + " --ntasks=" + str(nTasks) + "\n")
    if nTasks_per_node != -1:
        fout.write(sbatch + " --ntasks-per-node=" + str(nTasks_per_node) + "\n")
    if cpus_per_task != -1:
        fout.write(sbatch + " --cpus-per-task=" + str(cpus_per_task) + "\n")

    fout.write(sbatch + " -o " + str(fname_outfile) + "\n")

    #WRITE COMMAND
    fout.write(command)

    #Command to make shell executable
    makeprogram = "chmod u+x " + fname_shell
    os.system(makeprogram)
    return -1

def submit_job(fname_sh):
    sbatch = "sbatch"
    command = sbatch + " " + fname_sh
    os.system(command)