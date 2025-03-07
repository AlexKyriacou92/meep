import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d
import subprocess
#from GA_operators import flat_mutation, gaussian_mutation, clone, cross_breed
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

if 'TASKS' in cluster_settings.keys():
    TASKS = int(cluster_settings['TASKS'])
else:
    TASKS = 'NONE'

if 'NCPUS' in cluster_settings.keys():
    NCPUS = int(cluster_settings['NCPUS'])
else:
    NCPUS = 'NONE'

def make_job(fname_shell, fname_outfile, jobname, command, nNodes_min=NODES_MIN, nNodes_max=NODES_MAX,
             partition=PARTITION, days=DAYS, hours=HOURS, nodeMemory=MEMORY, tasks = TASKS, ncpus=NCPUS, account='NONE', email='NONE'):
    '''
    Function creates a shell file to run 1 job on Pleaides

    Arguments
    fname_shell : name of shell script <path/to/file.sh>
    fname_output : name of output file (where print statements go) <path/to/file.out>
    jobname : label of job (can be anything but should be consistent with fname_shell and fname_outfile)
    command : python command that runs job

    nNodes_min : minimum number of nodes in Pleaides to run job on
    nNodes_max : maximum number of nodes
    partition : set partition location ('short', 'long' or 'normal'

    days : maximum number of days to run job (kill if it excess this + hours)
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
    if tasks != "NONE":
        fout.write(sbatch + " --ntasks=" + str(tasks) + "\n")
    if ncpus != "NONE":
        fout.write(sbatch + " --cpus-per-task=" + str(ncpus) + "\n")
    if account != "NONE":
        fout.write(sbatch + " --account=" + str(account) + "\n")
    if email != "NONE":
        fout.write(sbatch + " --mail-user=" + str(email) + "\n")
    fout.write(sbatch + " --mem-per-cpu=" + str(nodeMemory) + " # in MB\n")
    fout.write(sbatch + " -o " + str(fname_outfile) + "\n")
    fout.write(command)

    #Command to make shell executable
    makeprogram = "chmod u+x " + fname_shell
    os.system(makeprogram)
    return -1

def make_command(config_file, bscan_data_file, nprof_matrix_file, ii, jj):
    command = 'python runSim_nprofile.py ' + config_file + ' ' + bscan_data_file + ' ' + nprof_matrix_file + ' ' + str(ii) + ' ' + str(jj)
    return command

def make_command_data(config_file, bscan_data_file, nprof_matrix_file, ii, jj):
    command = 'python runSim_FT_data.py ' + config_file + ' ' + bscan_data_file + ' ' + nprof_matrix_file + ' ' + str(
        ii) + ' ' + str(jj)
    return command

def submit_job(fname_sh):
    sbatch = "sbatch"
    command = sbatch + " " + fname_sh
    os.system(command)

def test_job(prefix, config_file, bscan_data_file, nprof_matrix_file, gene, individual):
    nprof_h5 = h5py.File(nprof_matrix_file, 'r')
    nprof_matrix = np.array(nprof_h5.get('n_profile_matrix'))
    nprof_list = nprof_matrix[gene]
    nProf = len(nprof_list)
    nprof_h5.close()

    fname_joblist = prefix + '-joblist.txt'
    fout_joblist = open(fname_joblist, 'w')
    fout_joblist.write('Joblist ' + prefix + '\n')
    fout_joblist.write(
        prefix + '\t' + config_file + '\t' + bscan_data_file + '\t' + nprof_matrix_file + '\t' + str(gene) + '\n')
    fout_joblist.write('shell_file' + '\t' + 'output_file' + '\t' + 'prof_number' + '\n \n')
    command = make_command(config_file, bscan_data_file, nprof_matrix_file, gene, individual)
    jobname = 'job-' + str(individual)
    fname_shell = prefix + jobname + '.sh'
    fname_out = prefix + jobname + '.out'
    line = fname_shell + '\t' + fname_out + '\t' + str(individual) + '\n'
    fout_joblist.write(line)
    make_job(fname_shell, fname_out, jobname, command)
    return fname_shell


def test_job_data(prefix, config_file, bscan_data_file, nprof_matrix_file, gene, individual):
    nprof_h5 = h5py.File(nprof_matrix_file, 'r')
    nprof_matrix = np.array(nprof_h5.get('n_profile_matrix'))
    nprof_list = nprof_matrix[gene]
    nProf = len(nprof_list)
    nprof_h5.close()

    fname_joblist = prefix + '-joblist.txt'
    fout_joblist = open(fname_joblist, 'w')
    fout_joblist.write('Joblist ' + prefix + '\n')
    fout_joblist.write(
        prefix + '\t' + config_file + '\t' + bscan_data_file + '\t' + nprof_matrix_file + '\t' + str(gene) + '\n')
    fout_joblist.write('shell_file' + '\t' + 'output_file' + '\t' + 'prof_number' + '\n \n')
    command = make_command_data(config_file, bscan_data_file, nprof_matrix_file, gene, individual)
    jobname = 'job-' + str(individual)
    fname_shell = prefix + jobname + '.sh'
    fname_out = prefix + jobname + '.out'
    line = fname_shell + '\t' + fname_out + '\t' + str(individual) + '\n'
    fout_joblist.write(line)
    make_job(fname_shell, fname_out, jobname, command)
    return fname_shell

def run_jobs(prefix, config_file, bscan_data_file, nprof_matrix_file, gene):
    nprof_h5 = h5py.File(nprof_matrix_file, 'r')
    nprof_matrix = np.array(nprof_h5.get(nprof_h5.get('n_profile_matrix')))
    nprof_list = nprof_matrix[gene]
    nProf = len(nprof_list)
    nprof_h5.close()

    fname_joblist = prefix + '-joblist.txt'
    fout_joblist = open(fname_joblist,'w')
    fout_joblist.write('Joblist ' + prefix + '\n')
    fout_joblist.write(prefix + '\t' + config_file + '\t' + bscan_data_file + '\t' + nprof_matrix_file + '\t' + str(gene) + '\n')
    fout_joblist.write('shell_file' + '\t' + 'output_file' + '\t' + 'prof_number' + '\n \n')
    for i in range(nProf):
        command = make_command(config_file, bscan_data_file, nprof_matrix_file, gene, i)
        jobname = 'job-' + str(i)
        fname_shell = prefix + jobname + '.sh'
        fname_out = prefix + jobname + '.out'
        line = fname_shell + '\t' + fname_out + '\t' + str(i) + '\n'
        fout_joblist.write(line)
        make_job(fname_shell, fname_out, jobname, command)
        submit_job(fname_shell)
    fout_joblist.close()
    return fname_joblist
'''
def make_sbatch_list(h5_file, config_file, prof_list):
    nProfs = len(prof_list)
    for i in range(nProfs):
        jobname = 'job' + str(i)
        fname_shell = jobname + '.sh'
        fname_out = jobname + '.out'
        make_command(h5_file, config_file, prof_list)
'''
def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output
