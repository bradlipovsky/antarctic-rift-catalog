import numpy as np
import matplotlib.pyplot as plt
#import os
#import h5py
#import dateutil.parser as dparser
#import time as t
#import numpy as np
#from pyproj import Transformer
#import pickle
import pandas as pd
#from netCDF4 import Dataset
#from scipy.spatial import cKDTree
#import matplotlib.pyplot as plt
#from multiprocessing import Pool
#from functools import partial
#import warnings

#import pyTMD.time
#from pyTMD.read_tide_model import extract_tidal_constants
#from pyTMD.predict_tidal_ts import predict_tidal_ts
#from pyTMD.predict_tide import predict_tide
#from pyTMD.infer_minor_corrections import infer_minor_corrections

#--------------------------------------------------------------------

def convert_to_centroid(rift_list,x,y):
    centroid_x = list()
    centroid_y = list()
    width = list()
    
    for r in rift_list:
        centroid_x.append( (x[r[0]] + x[r[1]-1])/2 )
        centroid_y.append( (y[r[0]] + y[r[1]-1])/2 )
        width.append( np.sqrt((x[r[0]] - x[r[1]-1])**2 + (y[r[0]] - y[r[1]-1])**2) )
        
    rift_data = {
        "x_centroid": centroid_x,
        "y_centroid": centroid_y,
        "width": width
    }
        
    return rift_data

#--------------------------------------------------------------------


'''
this is what I should be using
'''

def ash_get_rifts(atl06_data):
    '''
    arc.get_rifts 
    
    INPUT: atl06_data, dataframe with each "row" being the data from a single ATL06 file, 
            masked to the Antarctic Ice Shelves. The keys of the dictionary are defined in arc.ingest().
            
    OUTPUT: rift_obs, also a dictionary of lists, with each "row" corresponding to a single rift observation.  
            The dictionary keys are defined below.
    '''


    rift_obs = {
        "x_centroid": [],
        "y_centroid": [],
        "width": [],
        "time": [],
        "rgt": [],
        "azimuth": [],
        "sigma": [],
        "h":[],
        "beam":[],
        "data_row":[]
    }


    ttstart = t.perf_counter()

    for i, row in atl06_data.iterrows():
        
        if len(row["quality"]) == 0:
            continue

        # Only allow a certain percentage of data to be problematic
        percent_low_quality = sum( row["quality"]==1 )  / len(row["quality"])
        if percent_low_quality > 0.2:
            continue

        # Data product is posted at 20m.  Allowing spacing to be up to 25m allows for some missing data but not much.
        spacing = (max(row['x_atc']) - min(row['x_atc'])) / len(row['h'])
        if spacing > 25:
            continue
            

        
        # measure height relative to GEOID
        rift_list = ash_find_the_rifts( row['h'] - row['geoid'] - row['tides'])
        
        
        if len(rift_list) > 0:
            
            rift_azi = []
            rift_sig = []
            rift_h = []
            
            warnings.filterwarnings("error") # Treat warnings as errors just for this step 
            try:
                for rift_coords in rift_list:
                    rift_azi.append ( row['azimuth'][rift_coords[0]:rift_coords[1]].mean()   )
                    rift_sig.append ( row['h_sig'][rift_coords[0]:rift_coords[1]].mean()  )
                    rift_h.append   ( row['h'][rift_coords[0]:rift_coords[1]].mean() - \
                                      row['geoid'][rift_coords[0]:rift_coords[1]].mean())
            except RuntimeWarning:
                print('Abandoning this rift measurement (likely overflow from bad datapoint)')
                continue
            warnings.filterwarnings("default")
            
            output = convert_to_centroid(rift_list,row['x'],row['y'])
            output=pd.DataFrame(output)
            
            
            rift_obs['x_centroid'].extend( output['x_centroid'] )
            rift_obs['y_centroid'].extend( output['y_centroid'] )
            rift_obs['width'].extend( output['width'] )

            rift_obs['time'].extend( [ row['time'] ] * len(output) )
            rift_obs['rgt'].extend( [ row['rgt'] ] * len(output) )
            rift_obs['beam'].extend( [ row['beam'] ] * len(output) )
            rift_obs['data_row'].extend( [i] * len(output) )

            rift_obs['azimuth'].extend ( rift_azi )
            rift_obs['sigma'].extend( rift_sig )
            rift_obs['h'].extend( rift_h )
            

    # Save centroid locations in lat-lon
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
    if len(rift_obs['x_centroid'])>0:
        [lon,lat] = transformer.transform( rift_obs['x_centroid'] , rift_obs['y_centroid']  )
        rift_obs['lat'] = lat
        rift_obs['lon'] = lon

    ttend = t.perf_counter()
    print(' ')
    print('Found %i rifts.'%len(rift_obs["width"]))
    print('Time to detect rifts:', ttend - ttstart)
    
    return rift_obs



#------------------------

#def rift_cataloger(atl06_data)
    '''
    INPUT: atl06_data, dataframe with each "row" being the data from a single ATL06 file, 
            masked to the Antarctic Ice Shelves. The keys of the dictionary are defined in arc.ingest().
            
    OUTPUT: rift_obs, also a dictionary of lists, with each "row" corresponding to a single rift observation.  
            The dictionary keys are defined below.
    '''
'''
    rift_obs = {
        "x_centroid": [],
        "y_centroid": [],
        "width": [],
        "time": [],
        "rgt": [],
        "azimuth": [],
        "sigma": [],
        "h":[],
        "beam":[],
        "data_row":[],
        "confidence":[]
    } 
    
    #ttstart = t.perf_counter()

    for i, row in atl06_data.iterrows():
        
        if len(row["quality"]) == 0:
            continue

        # Only allow a certain percentage of data to be problematic
        percent_low_quality = sum( row["quality"]==1 )  / len(row["quality"])
        if percent_low_quality > 0.2:
            continue

        # Data product is posted at 20m.  Allowing spacing to be up to 25m allows for some missing data but not much.
        spacing = (max(row['x_atc']) - min(row['x_atc'])) / len(row['h'])
        if spacing > 25:
            continue
        
        # pass trace to rift detector
        # INPUTS (mandatory): h, x, y
        # INPUTS (optional): running mean half width (m), threshold (fraction of running mean >0, <1)
        
        # measure height relative to GEOID
        rift_list = rift_detector(row['h']-row['geoid']-row['tides'], row["x"], row["y"])
        # measure height relative to sea level
        #rift_list = rift_detector(row['h']-row['geoid']-row['mdt']-row['tides'], row["x"], row["y"])
        
        if len(rift_list) > 0:
'''    
    
    
    


def rift_detector(trace,x,y,run_mean_dist=10000,threshold=0.5):
    '''
    Find_the_rifts(trace) where trace is an array of surface heights
    returns a list of indices into trace ((start1,stop1),...
    where all consecutive values between trace[starti:stopi] are less
    than a threshold value
    
    rift_detector(trace,running_mean_distance,threshold)
    
    INPUT: 1) Array of freeboard heights
              (ICESat-2 measured height - Geoid - MDT - tides)
           2) x values
           3) y values
           4) Running mean distance either side (default = 10000 m)
           5) Threshold fraction of running mean height
              below which height points are considered rifts
              >0 to <1 (default = 0.5) 
               
    OUTPUT: 1) List of indices in trace ((start1,stop),...
               where all consecutive values between trace[starti:stopi]
               satisfy the threshold for rift detection
    ''' 
    # convert x and y to distance along track (in m)
    d = np.sqrt(x**2 + y**2)
    
    # calculate average separation of atl06
    # and number of points to calculate running mean over
    spacing = (max(d) - min(d)) / len(d)
    run_mean_nr = round(run_mean_dist / spacing)
    
    # Running mean
    # Could extend using one-sided? or expand ice shelf masks to grounded ice/ocean
    #h_run = running_mean(trace,(2*run_mean_nr)+1)
    #h_run = np.concatenate((np.zeros(run_mean_nr),h_run,np.zeros(run_mean_nr)))
    h_run = pd.DataFrame(trace).rolling((2*run_mean_nr)+1, min_periods=1, center=True).mean()
    h_run=h_run.values.flatten()
    
    # Rifts are defined as regions that are below the threshold fraction of running mean height
    # could add additional requirements
    # can use this to solve the masking problem, but it divides up rifts if there is masked points within them
    #in_rift = np.ma.where(trace < (threshold * h_run))
    in_rift = np.where(trace < (threshold * h_run))
    in_rift = in_rift[0]
    
    start = np.nan
    stop = np.nan
    segments=[]
    
    # Determine the first rift point in the list. Make sure that two rift walls are  
    # captured in the data.
    for i in range(len(in_rift)):
        if any(trace[0:in_rift[i]] > (threshold * h_run[i])):
            start = in_rift[i]
            break

    
    # now create a list with the format ((seg1start,seg1stop),(seg2start,seg2stop),...)
    # loop through all of the atl06 surface height points near sea level....
    for i in range(len(in_rift)):    
            
        if i == len(in_rift)-1:
            # This is the last point in the list. As before, make sure it is not the last 
            # point in the entire trace.
            if in_rift[i] < len(trace):
                stop = in_rift[i]
                if start<stop:            # Enforce that rift width must be greater than zero.
                    segments.append((start,stop))
                return segments
        
        if in_rift[i+1] > in_rift[i]+1:
            # This condition means that the next point in the list is not continuous. We 
            # interpret this to mean that we found the rift wall.
            
            stop = in_rift[i]+1
            if start<stop:            # Enforce that rift width must be greater than zero.
                segments.append((start,stop))
            start = in_rift[i+1]
            stop = np.nan

    return segments
    














