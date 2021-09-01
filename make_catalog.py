import numpy as np
import os
import arc
import importlib
importlib.reload(arc)
import json
import pandas as pd
import pickle

#shelf_names = ('brunt', 'fimbul', 'amery', 
#                'ap', 'ross', 'ronne', 'amundsen', 'east')

shelf_names = ( 'ronne', 'amundsen', 'east')

for shelf in shelf_names:
    
    print(' ================= %s ================= '%shelf)
    
    atl06_file_name = './atl06/' + shelf + '.pkl'
    atl06_filelist = './filelists/' + shelf + '-list.json'
    dataset_path = '/data/fast0/'

    with open(atl06_filelist,'rb') as handle:
        filelist = json.load(handle)
        
    arc.ingest(filelist,atl06_file_name,dataset_path)

    # Load data (deserialize)
    with open(atl06_file_name, 'rb') as handle:
        atl06_data = pickle.load(handle)

    # Find the rifts
    rift_obs = arc.get_rifts(atl06_data)

    # Store the rifts in a dataframe
    rift_obs=pd.DataFrame(rift_obs)
    
    rift_obs_output_file_name = 'rift_obs/' + shelf + '.pkl'
    
    with open(rift_obs_output_file_name, 'wb') as handle:
        pickle.dump(rift_obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
