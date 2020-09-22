import os
import h5py
import dateutil.parser as dparser
from datetime import datetime
import numpy as np
from pyproj import Transformer
import pickle
import pandas as pd

def ingest(data_directory,output_file_name):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031")
    
    ttstart = datetime.now()

    file_list = os.listdir(data_directory)
    files = [f for f in file_list if f.endswith('.h5')]

    print("Found %i files in the provided directory"%len(files))
    
    atl06_data = {"lat":list(),"lon":list(),"h":list(),"azimuth":list(),
                  "h_sig":list(),"rgt":list(),"time":list(), #"acquisition_number":list(),
                  "x":list(), "y":list(), "beam":list(), "quality":list(), "x_atc":list(), "geoid":list() }

    nf = len(files)
    for f in files:
        FILE_NAME = os.path.join(data_directory,f)
        fid = h5py.File(FILE_NAME, mode='r')

        for lr in ("l","r"):
            for i in range(1,4):
                try:
                    h_xatc = fid['gt%i%s/land_ice_segments/ground_track/x_atc'%(i,lr)][:]
                    h_li = fid['gt%i%s/land_ice_segments/h_li'%(i,lr)][:]
                    h_lat = fid['gt%i%s/land_ice_segments/latitude'%(i,lr)][:]
                    h_lon = fid['gt%i%s/land_ice_segments/longitude'%(i,lr)][:]
                    h_li_sigma = fid['gt%i%s/land_ice_segments/h_li_sigma'%(i,lr)][:]
                    seg_az = fid['gt%i%s/land_ice_segments/ground_track/seg_azimuth'%(i,lr)][:]
                    rgt = fid['/orbit_info/rgt'][0]
                    quality = fid['gt%i%s/land_ice_segments/atl06_quality_summary'%(i,lr)][:]
                    time = dparser.parse( fid['/ancillary_data/data_start_utc'][0] ,fuzzy=True )
                    beam = "%i%s"%(i,lr)
                    geoid = fid['/gt%i%s/land_ice_segments/dem/geoid_h'%(i,lr)][:]
                    [h_x,h_y] = transformer.transform( h_lat , h_lon )

                except KeyError:
        #                 print("wtf key error")
                    continue

#             This is just used for Brunt:
#                         Not clear why some of the data is out of the region of interest
                if any(h_lon>0):
                    continue

                atl06_data["lat"].append( h_lat )
                atl06_data["lon"].append( h_lon )
                atl06_data["x"].append( h_x )
                atl06_data["y"].append( h_y )
                atl06_data["h_sig"].append( h_li_sigma )
                atl06_data["h"].append( h_li)
                atl06_data["azimuth"].append( seg_az )
                atl06_data["rgt"].append( rgt )
                atl06_data["time"].append( time )
                atl06_data["beam"].append ( beam )
                atl06_data["quality"].append ( quality )
                atl06_data["x_atc"].append( h_xatc )
                atl06_data["geoid"].append( geoid )
                
        fid.close()

    ttend = datetime.now()
    print('Time to read the H5 files: ', ttend - ttstart)
    
    #
    # Now convert to polar stereo coordinates
    #
    
    


    # Store data (serialize)
    with open(output_file_name, 'wb') as handle:
        pickle.dump(atl06_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
        
        
def find_the_rifts(trace):
    #
    # find_the_rifts(trace) where trace is an array of surface heights
    # returns a list of indices into trace ((start1,stop1),...
    # where all consecutive values between trace[starti:stopi] are less
    # than a threshold value
    #
    
    # Threshold height for detection as a rift
    upper_thr = 5
    lower_thr = -10
    
    # Rifts are defined as regions that are near 0m above sea level
    # recall that some erroneous points in ATL06 are large and negative
    in_rift = np.where( (trace < upper_thr) & (trace>lower_thr) ) 
    in_rift = in_rift[0]
    
    # Don't trust 1-point measurements
    if len(in_rift) < 2:
        return []
    
    start = np.nan
    stop = np.nan
    segments=[]
    
    # Determine the first rift point in the list. Make sure that two rift walls are  
    # captured in the data.
    for i in range(len(in_rift)):
        if any(trace[0:in_rift[i]] > upper_thr):
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
#             print("Added {strt} , {stp}".format(strt=start,stp=stop))
            start = in_rift[i+1]
            stop = np.nan
            
    return segments
        
    
    
    
    
    
def convert_to_centroid(rift_list,x,y):
    centroid_x = list()
    centroid_y = list()
    width = list()
    
    for r in rift_list:
        centroid_x.append( (x[r[0]] + x[r[1]-1])/2 )
        centroid_y.append( (y[r[0]] + y[r[1]-1])/2 )
        width.append( np.sqrt((x[r[0]] - x[r[1]-1])**2 + (y[r[0]] - y[r[1]-1])**2) )
        
    rift_data = {
        "x-centroid": centroid_x,
        "y-centroid": centroid_y,
        "width": width
    }
        
    return rift_data






def get_rifts(atl06_data):

    atl06_dataframe = pd.DataFrame(atl06_data)

    rift_obs = {
        "x-centroid": [],
        "y-centroid": [],
        "width": [],
        "time": [],
        "rgt": [],
        "azimuth": [],
        "sigma": [],
        "h":[],
        "beam":[],
        "data_row":[]
    }


    ttstart = datetime.now()

    for i, row in atl06_dataframe.iterrows():
        
        # Check to see how much data is available in this aquisition.  
        # Do this by checking to see what data is in a physically reasonable range for ice shelves, say -10 to 200m asl
        # For typical data, this is often between 99% and 100%
#         h=row["h"]
#         data_in_physical_range = len(h[(h<200)&(h>-10)])/len(h)
#         availability_threshold = 0.95
#         if data_in_physical_range < availability_threshold:
#             continue

        # measure height relative to GEOID
        these_heights = row['h'] - row['geoid']
        
        # Data product is posted at 20m.  Allowing spacing to be up to 25m allows for some missing data but not much.
        spacing = (max(row['x_atc']) - min(row['x_atc'])) / len(row['h'])
        if spacing > 25:
            continue
            
        # Only allow a certain percentage of data to be problematic
        percent_low_quality = sum(row["quality"]==1)  / len(row["quality"])
        if percent_low_quality > 0.2:
            continue
        
        rift_list = find_the_rifts( these_heights )
        
        if len(rift_list) > 0:
            
            rift_azi = []
            rift_sig = []
            rift_h = []
            for rift_coords in rift_list:
                rift_azi.append ( row['azimuth'][rift_coords[0]:rift_coords[1]].mean()   )
                rift_sig.append ( row['h_sig'][rift_coords[0]:rift_coords[1]].mean()  )
                rift_h.append   ( row['h'][rift_coords[0]:rift_coords[1]].mean()  )
            
            output = convert_to_centroid(rift_list,row['x'],row['y'])
            output=pd.DataFrame(output)
            
            rift_obs['x-centroid'].extend( output['x-centroid'] )
            rift_obs['y-centroid'].extend( output['y-centroid'] )
            rift_obs['width'].extend( output['width'] )
            
            rift_obs['time'].extend( [ row['time'] ] * len(output) )
            rift_obs['rgt'].extend( [ row['rgt'] ] * len(output) )
            rift_obs['beam'].extend( [ row['beam'] ] * len(output) )
            rift_obs['data_row'].extend( [i] * len(output) )
            
            rift_obs['azimuth'].extend ( rift_azi )
            rift_obs['sigma'].extend( rift_sig )
            rift_obs['h'].extend( rift_h )
            
#             print("len(rift_list) = %f, len(output['x-centroid']) = %f"%(len(rift_list),len(output['x-centroid'])))
            
#         print('Completed %i of %i at %s'%(i,na,datetime.now()))

    # Save centroid locations in lat-lon
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
    if len(rift_obs['x-centroid'])>0:
        [lon,lat] = transformer.transform( rift_obs['x-centroid'] , rift_obs['y-centroid']  )
        rift_obs['lat'] = lat
        rift_obs['lon'] = lon

    ttend = datetime.now()
    print(' ')
    print('Found %i rifts.'%len(rift_obs["width"]))
    print('Time to detect rifts:', ttend - ttstart)
    
    return rift_obs
