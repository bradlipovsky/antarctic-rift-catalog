import numpy as np
import os
import arc
import importlib
importlib.reload(arc)
import json

shelf_name='brunt-riiser-ekstrom'

atl06_file_name = './atl06_' + shelf_name + '-atl06.pkl'
atl06_filelist = './filelists/' + shelf_name + '-list.json'
dataset_path = '/data/fast0/'

with open(atl06_filelist,'rb') as handle:
    filelist = json.load(handle)
arc.ingest(filelist,atl06_file_name,dataset_path)


