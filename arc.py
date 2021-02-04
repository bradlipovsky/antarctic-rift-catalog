import os
import h5py
import dateutil.parser as dparser
import time as t
import numpy as np
from pyproj import Transformer
import pickle
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

def ingest(file_list,output_file_name, maskfile):
    '''
    Organize a bunch of ATL06 H5 files and save the result.  Rejects all non-ice shelf data points.
    
    file_list is the list of files.  If the files are stored in an AWS S3 bucket then they will be copied to the 
        local directory, read, and then discarded.
        
    output_file_name is the name of the pickle file where the output structure is stored
    
    maskfile is an iceshelf mask
    '''
    
    # Load BedMachine ice mask.  We use KD-Trees to do a nearest-neighbor search. The cKDTree takes about 90s to generate.  
    # Check to see if this is already done... much faster to make it once and then load it everytime it's needed
    print('Working on the mask...')
    ttstart=t.perf_counter()
    fh = Dataset(maskfile, mode='r')
    x = fh.variables['x'][:]
    y = np.flipud(fh.variables['y'][:])
    mask = np.flipud(fh.variables['mask'][:]) # 0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice
    
    kdt_file = 'bma-v2-ckdt.pickle'
    if os.path.exists(kdt_file):
        with open(kdt_file, 'rb') as handle:
            tree = pickle.load(handle)
    else:
        (xm,ym) = np.meshgrid(x,y)
        tree = cKDTree( np.column_stack((xm.flatten(), ym.flatten())) )    
        pickle.dump( tree, open(kdt_file,'wb') )
    print('     Mask done after %f s'%(t.perf_counter()-ttstart))
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031")
    
    ttstart = t.perf_counter()
    t0 = ttstart
    
    atl06_data = {"lat":list(),"lon":list(),"h":list(),"azimuth":list(),
                  "h_sig":list(),"rgt":list(),"time":list(), #"acquisition_number":list(),
                  "x":list(), "y":list(), "beam":list(), "quality":list(), "x_atc":list(), "geoid":list() }
    
    if any(f.startswith('s3') for f in file_list):
        import s3fs
    
    starting_time_for_files = t.perf_counter()
    for f,i in zip(file_list,range(0,len(file_list))):
        if i>0:
            percent_finished=i/len(file_list)
            dt = t.perf_counter()-starting_time_for_files
            wt =  t.perf_counter() - t0
            time_remaining = dt/percent_finished - wt
            print('%f percent done.  Remaining time is %f s. Wall time is %f s.'%(100*percent_finished,time_remaining,wt))
        if f.startswith('s3'):
            s3 = s3fs.S3FileSystem(anon=True)
            print('Moving File %d S3->EBS ...'%i)
            open_this = './temp.h5'
            try:
                s3.get(f, open_this)
            except FileNotFoundError:
                print('WARNING:  FILE NOT FOUND')
                continue
            
            ttend = t.perf_counter()
            print('     Time to move file: ', ttend - ttstart)
            ttstart = t.perf_counter()
            
        else:
            open_this = f
            print('     Opened local file.')
            
        fid = h5py.File(open_this, mode='r')

        for lr in ("l","r"):
            for i in range(1,4):
                try:
                    h_xatc =     np.array(fid['gt%i%s/land_ice_segments/ground_track/x_atc'%(i,lr)][:])
                    h_li =       np.array(fid['gt%i%s/land_ice_segments/h_li'%(i,lr)][:])
                    h_lat =      np.array(fid['gt%i%s/land_ice_segments/latitude'%(i,lr)][:])
                    h_lon =      np.array(fid['gt%i%s/land_ice_segments/longitude'%(i,lr)][:])
                    h_li_sigma = np.array(fid['gt%i%s/land_ice_segments/h_li_sigma'%(i,lr)][:])
                    seg_az =     np.array(fid['gt%i%s/land_ice_segments/ground_track/seg_azimuth'%(i,lr)][:])
                    quality =    np.array(fid['gt%i%s/land_ice_segments/atl06_quality_summary'%(i,lr)][:])
                    geoid =      np.array(fid['/gt%i%s/land_ice_segments/dem/geoid_h'%(i,lr)][:])
                    
                    rgt = fid['/orbit_info/rgt'][0]
                    time = dparser.parse( fid['/ancillary_data/data_start_utc'][0] ,fuzzy=True )
                    beam = "%i%s"%(i,lr)
                    
                    [h_x,h_y] = transformer.transform( h_lat , h_lon )
                    
                except KeyError:
                    print("wtf key error")
                    continue

#             This is just used for Brunt:
#                         Not clear why some of the data is out of the region of interest
#                 if any(h_lon>0):
#                     continue

                # Only add the point if is in the ice shelf mask
                nothing, inds = this_mask = tree.query(np.column_stack((h_x,h_y)), k = 1)
                this_mask = np.array(mask.flatten()[inds])
            
#                 atl06_data["lat"].append( h_lat )
#                 atl06_data["lon"].append( h_lon )
#                 atl06_data["x"].append( h_x )
#                 atl06_data["y"].append( h_y )
#                 atl06_data["h"].append( h_li )
#                 atl06_data["h_sig"].append( h_li_sigma )
#                 atl06_data["azimuth"].append( seg_az )
#                 atl06_data["quality"].append( quality )
#                 atl06_data["x_atc"].append( h_xatc )
#                 atl06_data["geoid"].append( geoid )
            
                atl06_data["lat"].append( h_lat[np.where(this_mask==3)] )
                atl06_data["lon"].append( h_lon[np.where(this_mask==3)] )
                atl06_data["x"].append( h_x[np.where(this_mask==3)] )
                atl06_data["y"].append( h_y[np.where(this_mask==3)] )
                atl06_data["h"].append( h_li[np.where(this_mask==3)] )
                atl06_data["h_sig"].append( h_li_sigma[np.where(this_mask==3)] )
                atl06_data["azimuth"].append( seg_az[np.where(this_mask==3)] )
                atl06_data["quality"].append( quality[np.where(this_mask==3)] )
                atl06_data["x_atc"].append( h_xatc[np.where(this_mask==3)] )
                atl06_data["geoid"].append( geoid[np.where(this_mask==3)] )
                
                atl06_data["rgt"].append( rgt )
                atl06_data["time"].append( time )
                atl06_data["beam"].append ( beam )
            
#                 atl06_data["lat"].append( np.array([h_lat[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["lon"].append( np.array([h_lon[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["x"].append( np.array([h_x[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["y"].append( np.array([h_y[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["h_sig"].append( np.array([h_li[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["h"].append( np.array([h_li[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["azimuth"].append( np.array([seg_az[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["quality"].append ( np.array([quality[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["x_atc"].append( np.array([h_xatc[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
#                 atl06_data["geoid"].append( np.array([geoid[i] for i in range(len(h_li)) if this_mask[i] > 2] ) )
                
        ttend = t.perf_counter()
        print('     Time to process file: ', ttend - ttstart)
        ttstart = t.perf_counter()
                
        fid.close()
        if f.startswith('s3'):
            os.remove('./temp.h5')

    ttend = t.perf_counter()
    print('Time to read the H5 files: ', ttend - ttstart)

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
    '''
    arc.get_rifts 
    
    INPUT: atl06_data, the dictionary of lists of ATL06 data (this format can easily be converted to a pandas dataframe).
            Each "row" contains the data from a single ATL06 file, masked to the Antarctic Ice Shelves.
            The keys of the dictionary are defined in arc.ingest().
            
    OUTPUT: rift_obs, also a dictionary of lists, with each "row" corresponding to a single rift observation.  
            The dictionary keys are defined below.
    '''

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


    ttstart = t.perf_counter()

    for i, row in atl06_dataframe.iterrows():
        
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
        rift_list = find_the_rifts( row['h'] - row['geoid'] )
        
        
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
            

    # Save centroid locations in lat-lon
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
    if len(rift_obs['x-centroid'])>0:
        [lon,lat] = transformer.transform( rift_obs['x-centroid'] , rift_obs['y-centroid']  )
        rift_obs['lat'] = lat
        rift_obs['lon'] = lon

    ttend = t.perf_counter()
    print(' ')
    print('Found %i rifts.'%len(rift_obs["width"]))
    print('Time to detect rifts:', ttend - ttstart)
    
    return rift_obs
