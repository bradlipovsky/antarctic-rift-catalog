import numpy as np
import os
import arc
import importlib
importlib.reload(arc)
import json
import pandas as pd
import pickle

def main():
    atl06_path = "/data/fast1/arc/atl06"
    filelist_path = "/data/fast1/arc/filelists"
    dataset_path = '/data/fast0/'
    output_path = '/data/fast1/arc/rift_obs'

    #shelf_names = ['brunt', 'fimbul', 'amery', 
    #    'ap', 'ross', 'ronne', 'amundsen', 'east']

    # shelf_names = ['ross', 'ronne', 'amundsen', 'east']

    shelf_names = ['brunt']

    for shelf in shelf_names:


        print(" ")
        print('==================================')
        print(' PROCESSING THE ATL06 DATA (%s)'%shelf)
        print('==================================')

        atl06_file_name = os.path.join(atl06_path, shelf + '.pkl')
        atl06_filelist = os.path.join(filelist_path, shelf + '-list.json')

        with open(atl06_filelist,'rb') as handle:
            filelist = json.load(handle)

        arc.ingest(filelist,atl06_file_name,dataset_path)

        # Load data
        with open(atl06_file_name, 'rb') as handle:
            atl06_data = pickle.load(handle)

        print('==================================')
        print(' FINDING THE RIFTS (%s)'%shelf)
        print('==================================')
        # Find the rifts
        rift_obs = arc.get_rifts(atl06_data)

        # Store the rifts in a dataframe
        rift_obs=pd.DataFrame(rift_obs)

        rift_obs_output_file_name = os.path.join(output_path, shelf + '.pkl')

        with open(rift_obs_output_file_name, 'wb') as handle:
            pickle.dump(rift_obs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
