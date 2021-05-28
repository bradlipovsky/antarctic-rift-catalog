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
from multiprocessing import Pool
from functools import partial

import pyTMD.time
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.predict_tide import predict_tide
from pyTMD.infer_minor_corrections import infer_minor_corrections


def ingest(file_list,output_file_name, datapath,verbose=False):
    '''
    Organize a bunch of ATL06 H5 files and save the result.  Rejects all non-ice shelf data points.
    
    file_list is the list of files.  If the files are stored in an AWS S3 bucket then they will be copied to the 
        local directory, read, and then discarded.
        
    output_file_name is the name of the file where the output structure is stored
    
    maskfile is an iceshelf mask
    '''
    dataset_path = datapath + 'datasets/'
    maskfile = 'BedMachineAntarctica_2020-07-15_v02.nc'
    kdt_file = 'BedMachine2-ckdt.pkl'
    
    '''
    Does the output file already exist?
    '''
    if os.path.isfile(output_file_name):
        print("Data already saved, so there's no need to ingest data. \
    To repeat the data ingest, it would probably be best to change the filename of the \
    existing file.")
        return
    
    '''
    Load BedMachine ice mask.  We use KD-Trees to do a nearest-neighbor search. 
    The cKDTree takes about 90s to generate... much faster to make it once and 
    then load it everytime it's needed
    '''
    print('Working on the mask...')
    ttstart=t.perf_counter()
    fh = Dataset(dataset_path+maskfile, mode='r')
    x = fh.variables['x'][:]
    y = np.flipud(fh.variables['y'][:])
    
    # 0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice
    mask = np.flipud(fh.variables['mask'][:]) 
    
    if os.path.exists(dataset_path+kdt_file):
        print('     Loading existing ckdt.')
        with open(dataset_path+kdt_file,'rb') as handle:
            tree = pickle.load(handle)
            
    else:
        print('     Constructing new ckdt.')
        (xm,ym) = np.meshgrid(x,y)
        tree = cKDTree( np.column_stack((xm.flatten(), ym.flatten())) )    
        pickle.dump( tree, open(dataset_path+kdt_file,'wb') )
    print('     Mask loaded after %f s'%(t.perf_counter()-ttstart))
    

    '''
    Read all the files in parallel.  Note that partial can't accept any arguments that can't be pickled.
    For this reason, we have to do the mask calculations later (the KD-Tree can't be pickled.
    '''
    ttstart = t.perf_counter()
    func = partial(load_one_file, datapath,verbose)
    nproc = 8
    with Pool(nproc) as p:
        atl06_data = p.map(func, file_list)
    df = pd.DataFrame(atl06_data)
    print('Time to read the H5 files: ', t.perf_counter() - ttstart)

    '''
    Delete segments with less than ten data points
    '''
    df = df[ df['h'].map(np.size) >= 10]
    
    '''
    Apply the ice mask
    '''
    ttstart = t.perf_counter()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031")
    df_mask = df.apply (lambda row: apply_mask(row,transformer,tree,mask), axis=1)
    output = pd.DataFrame(list(df_mask))
    output.dropna(inplace=True)
    print('Time to apply ice shelf mask: ', t.perf_counter() - ttstart)
    
    '''
    Calculate tides
    '''
    unpanda = output.to_dict('records')
    ttstart = t.perf_counter()
    nproc = 96
    func = partial(run_pyTMD, dataset_path)
    p = Pool(nproc)
    pool_results = p.map(func, unpanda)
    p.close()
    p.join()
    output['tides'] = pool_results
    print('Calculated tides in %f s.'%(t.perf_counter() - ttstart))

    '''
    Write to file
    '''
    ttstart = t.perf_counter()
    with open(output_file_name, 'wb') as handle:
        pickle.dump(output, handle)
    print('Time to save all of the data: ', t.perf_counter() - ttstart)
    
def run_pyTMD(TIDE_PATH,row):
    # LAT, LON can be vectors, TIME is a scalar
    # 0 < LON < 360
    # output is vector of tide height in meters
    
    LAT = row['lat']
    LON = row['lon']
    TIME = row['time']
#     print('.')
    if len(LAT) < 10:
        return []

    tide_dir = TIDE_PATH

    #-- calculate a weeks forecast every minute
#     seconds = TIME.second + np.arange(0,1)
    tide_time = pyTMD.time.convert_calendar_dates(TIME.year, 
                                                  TIME.month,
                                                  TIME.day, 
                                                  TIME.hour, 
                                                  TIME.minute, 
                                                  TIME.second)

    #-- delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])

    grid_file = os.path.join(tide_dir,'grid_CATS2008')
    model_file = os.path.join(tide_dir,'hf.CATS2008.out')
    model_format = 'OTIS'
    EPSG = 'CATS2008'
    TYPE = 'z'

    #-- read tidal constants and interpolate to grid points
#     for lo,la in zip(LON,LAT)
    amp,ph,D,c = extract_tidal_constants(np.atleast_1d(LON),np.atleast_1d(LAT),
        grid_file,model_file,EPSG,TYPE=TYPE,METHOD='spline',GRID=model_format)
    deltat = np.zeros_like(tide_time)

    #-- calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    #-- calculate constituent oscillation
    hc = amp*np.exp(cph)

    #-- convert time from MJD to days relative to Jan 1, 1992 (48622 MJD)
    #-- predict tidal elevations at time 1 and infer minor corrections

    TIDE = predict_tide(tide_time, hc, c,
        DELTAT=deltat, CORRECTIONS=model_format)
    MINOR = infer_minor_corrections(tide_time, hc, c,
        DELTAT=deltat, CORRECTIONS=model_format)

    TIDE.data[:] += MINOR.data[:]

    return TIDE




def get_coords(shelf_name):
    ''' This is just a holding bin of shelf polygons'
    '''
    
    # Format is WSNE
    switcher = {
        'brunt': [-27.8, -76.1, -0.001, -69.6], # Brunt-Riiser-Ekstrom System
        'fimbul': [0.001,-71.5, 39.5, -68.6],
        'amery': [67.6, -72.44,74.87,-68.39],
        'ap': [-83.5,-74.1,-54.2,-62.8],
        'ross': [159,-86,-147,-69],
        'ronne': [-80,-82,-28,-74.5],
        'amundsen':[-147,-75.5,-83.5,-71.5],
        'east':[80,-70,159,-64]
    }
        
    return switcher[shelf_name]


def apply_mask(row,transformer,tree,mask):
    '''
    Apply the ice shelf mask to each row.
    '''

    [h_x,h_y] = transformer.transform( row.lat , row.lon )
    if isinstance(h_x,float): 
        return {}
    nothing, inds = tree.query(np.column_stack((h_x,h_y)), k = 1, workers=2)
    this_mask = np.array(mask.flatten()[inds])
    
    new_row = {}
    new_row['x'] = h_x[np.where(this_mask==3)]
    new_row['y'] = h_y[np.where(this_mask==3)]
    for key in ("x_atc","h","lat","lon","azimuth","quality","geoid","h_sig"):
        new_row[key] = row[key][np.where(this_mask==3)]
    new_row['rgt'] = row['rgt']
    new_row['time'] = row['time']
    new_row['beam'] = row['beam']

    return new_row
    
def load_one_file(datapath,verbose,f):
    if verbose:
        print('     Opening local file %s'%f)
    try:
        fid = h5py.File(datapath + f, mode='r')
    except FileNotFoundError:
        print (     'ERROR: File not found,  %s'%f)
        return {}

    for lr in ("l","r"):
        for i in range(1,4):
            try:
                out = {}
                
                out["x_atc"] =  np.array(fid['gt%i%s/land_ice_segments/ground_track/x_atc'%(i,lr)][:])
                out["h"] =      np.array(fid['gt%i%s/land_ice_segments/h_li'%(i,lr)][:])
                out["lat"] =    np.array(fid['gt%i%s/land_ice_segments/latitude'%(i,lr)][:])
                out["lon"] =    np.array(fid['gt%i%s/land_ice_segments/longitude'%(i,lr)][:])
                out["h_sig"] =  np.array(fid['gt%i%s/land_ice_segments/h_li_sigma'%(i,lr)][:])
                out["azimuth"]= np.array(fid['gt%i%s/land_ice_segments/ground_track/seg_azimuth'%(i,lr)][:])
                out["quality"]= np.array(fid['gt%i%s/land_ice_segments/atl06_quality_summary'%(i,lr)][:])
                out["geoid"] =  np.array(fid['/gt%i%s/land_ice_segments/dem/geoid_h'%(i,lr)][:])

                out["rgt"] = fid['/orbit_info/rgt'][0]
                out["time"] = dparser.parse( fid['/ancillary_data/data_start_utc'][0] ,fuzzy=True )
                out["beam"] = "%i%s"%(i,lr)

            except KeyError:
                print("ERROR: Key error, %s"%f)

    fid.close()
    return out


        

        
        
        
        
        
        
def find_the_rifts(trace):
    '''
    Find_the_rifts(trace) where trace is an array of surface heights
    returns a list of indices into trace ((start1,stop1),...
    where all consecutive values between trace[starti:stopi] are less
    than a threshold value
    '''
    
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
    
    INPUT: atl06_data, dataframe with each "row" being the data from a single ATL06 file, 
            masked to the Antarctic Ice Shelves. The keys of the dictionary are defined in arc.ingest().
            
    OUTPUT: rift_obs, also a dictionary of lists, with each "row" corresponding to a single rift observation.  
            The dictionary keys are defined below.
    '''


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
        rift_list = find_the_rifts( row['h'] - row['geoid'] - row['tides'])
        
        
        if len(rift_list) > 0:
            
            rift_azi = []
            rift_sig = []
            rift_h = []
            for rift_coords in rift_list:
                rift_azi.append ( row['azimuth'][rift_coords[0]:rift_coords[1]].mean()   )
                rift_sig.append ( row['h_sig'][rift_coords[0]:rift_coords[1]].mean()  )
                rift_h.append   ( row['h'][rift_coords[0]:rift_coords[1]].mean() - \
                                  row['geoid'][rift_coords[0]:rift_coords[1]].mean())
            
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
