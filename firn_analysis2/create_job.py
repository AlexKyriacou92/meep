import sys
import osc_scripting
import os
import pathlib
import configparser

from osc_scripting import make_job

cluster_config = sys.argv[1]
config_path = pathlib.Path(__file__).parent.absolute() / cluster_config
config = configparser.ConfigParser()
print(config_path)
config.read(config_path)

#Cluster Settings
cluster_settings = config['CLUSTER']

NODES_MIN = int(cluster_settings['NODES_MIN'])
NODES_MAX = int(cluster_settings['NODES_MAX'])

PARTITION = cluster_settings['PARTITION']
DAYS = int(cluster_settings['DAYS'])
HOURS = int(cluster_settings['HOURS'])
MEMORY = int(cluster_settings['MEMORY']) # in MB
TASKS = int(cluster_settings['TASKS'])
NCPUS = int(cluster_settings['NCPUS'])

if 'ACCOUNT' in cluster_settings.keys(): #the account has to be specified by osc, check if memeber
    ACCOUNT=cluster_settings['ACCOUNT']
else:
    print('error, add ACCOUNT as a field to the cluster config', sys.argv[1])
    print('checking keys:', cluster_settings.keys())
    sys.exit()
if 'EMAIL' in cluster_settings.keys():
    EMAIL = cluster_settings['EMAIL']
else:
    EMAIL='NONE'

#print('TASKS=', TASKS, 'NPCUS=', NCPUS)

job_settings = config['JOB']
operator = job_settings['operator']
script = job_settings['script']
arg = job_settings['arg']
prefix = job_settings['prefix']
mpi = job_settings['mpi']
py_cmd = operator + ' ' + script + ' ' + arg
cmd = mpi + ' -n ' + str(TASKS) + ' ' + py_cmd

fname_shell = prefix + '.sh'
fname_out = prefix + '.out'
job_name = prefix
make_job(fname_shell=fname_shell, fname_outfile=fname_out, jobname=job_name, command=cmd,
         nNodes_min=NODES_MIN, nNodes_max=NODES_MAX, partition=PARTITION,
         days=DAYS, hours=HOURS, nodeMemory=MEMORY, tasks = TASKS, ncpus=NCPUS, account=ACCOUNT, email=EMAIL)
