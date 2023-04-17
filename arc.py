import os
import h5py
import dateutil.parser as dparser
import time as t
import numpy as np
from pyproj import Transformer
import pickle
import pandas as pd
from netCDF4 import Dataset
import scipy
from scipy.spatial import cKDTree
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from multiprocessing import Pool
from functools import partial
import warnings
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
        print("Data already saved, so there's no need to ingest data.")
        print("To repeat the data ingest, it would probably be best")
        print("to change the filename of the existing file.")
        print(" ")
        print("Current filename is: %s"%output_file_name)
        print(" ")
        val = input("DANGER ZONE: Type YES to overwrite:   ")
        if val!='YES':
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
    Read all the files in parallel.  Note that partial can't accept any arguments that can't 
    be pickled. For this reason, we have to do the mask calculations later (the KD-Tree can't 
    be pickled.
    '''
    ttstart = t.perf_counter()
    func = partial(load_one_file, datapath,verbose)
    nproc = 8
    with Pool(nproc) as p:
        atl06_data = p.map(func, file_list)
    
    '''
    Expand the p.map results.  
    
    atl06_data looks like this:
    len(atl06_data) = 2187  = number of p.map calls
    len(atl06_data[0]['lat']) = 6  = number of beams
    
    list of dictionaries with list elements -> list of dataframes -> single dataframe
    '''
    output_list = list(map(pd.DataFrame, atl06_data))
    df = pd.concat(output_list,ignore_index=True)
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
    
    apply_mask_partial = lambda x: apply_mask(transformer,tree,mask,x)
    df_mask = df.apply (apply_mask_partial, axis=1)
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


def apply_mask(transformer,tree,mask,row):
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
        fid = h5py.File( os.path.join(datapath, f), mode='r')
    except (FileNotFoundError, OSError) as e:
        print (     'ERROR: File not found,  %s'%f)
        return {}
    
    
    
    out = {"lat":list(),"lon":list(),"h":list(),"azimuth":list(),
                  "h_sig":list(),"rgt":list(),"time":list(), #"acquisition_number":list(),
                   "beam":list(), "quality":list(), "x_atc":list(),
                   "geoid":list() }
    
    for lr in ("l","r"):
        for i in range(1,4):
            try:
                base = 'gt%i%s/land_ice_segments/'%(i,lr)
                out["x_atc"].append(   np.array(fid[base + 'ground_track/x_atc'][:]) )
                out["h"].append(       np.array(fid[base + 'h_li'][:]) )
                out["lat"].append(     np.array(fid[base + 'latitude'][:]) )
                out["lon"].append(     np.array(fid[base + 'longitude'][:]) )
                out["h_sig"].append(   np.array(fid[base + 'h_li_sigma'][:]) )
                out["azimuth"].append( np.array(fid[base + 'ground_track/seg_azimuth'][:]) )
                out["quality"].append( np.array(fid[base + 'atl06_quality_summary'][:]) )
                out["geoid"].append(   np.array(fid[base + 'dem/geoid_h'][:]) )
                out["rgt"].append(  fid['/orbit_info/rgt'][0] )
                out["time"].append(  dparser.parse( \
                                       fid['/ancillary_data/data_start_utc'][0] ,fuzzy=True ) )
                out["beam"].append(  "%i%s"%(i,lr) )

            except KeyError as e:
                print(e)
                print("ERROR: Key error, %s"%f)

    fid.close()
    return out






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






def get_rifts(atl06_data):
    '''
    arc.get_rifts 
    
    INPUT: atl06_data, dataframe with each "row" being the data from a single ATL06 file, 
            masked to the Antarctic Ice Shelves. The keys of the dictionary are defined 
            in arc.ingest().
            
    OUTPUT: rift_obs, also a dictionary of lists, with each "row" corresponding to a 
            single rift observation.  
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

        # Data product is posted at 20m.  Allowing spacing to be up to 25m allows 
        # for some missing data but not much.
        spacing = (max(row['x_atc']) - min(row['x_atc'])) / len(row['h'])
        if spacing > 25:
            continue

        # Check if tides and geoid are avaialble
        if len(row['h']) == len(row['geoid']) and len(row['geoid']) == len(row['tides']):
        
            # measure height relative to GEOID
            rift_list = find_the_rifts( row['h'] - row['geoid'] - row['tides'])
            
        else:
            continue
        
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
            
            
def rift_detector(trace,trace_run,d,threshold=0.6):
    '''
    Find_the_rifts(trace) where trace is an array of surface heights
    returns a list of indices into trace ((start1,stop1),...
    where all consecutive values between trace[starti:stopi] are less
    than a threshold value
    
    rift_detector(trace,trace_running_mean,distance,threshold)
    
    INPUT: 1) Array of freeboard heights
              (ICESat-2 measured height - Geoid - MDT - tides)
           2) running mean through height data
              (ie smoothed surface). Must be same size
           3) distance along track
           5) Threshold fraction of running mean height
              below which height points are considered rifts
              >0 to <1 (default = 0.6) 
               
    OUTPUT: 1) List of indices in trace ((start1,stop),...
               where all consecutive values between trace[starti:stopi]
               satisfy the threshold for rift detection
    '''

    # Rifts are defined as regions that are below the threshold fraction of running mean height
    #in_rift = np.ma.where(trace < (threshold * trace_run))
    in_rift = np.where(trace < (threshold * trace_run))
    in_rift = in_rift[0]
    
    start = np.nan
    stop = np.nan
    segments=[]
    
    # Determine the first rift point in the list. Make sure that two rift walls are  
    # captured in the data.
    for i in range(len(in_rift)):
        if any(trace[0:in_rift[i]] > (threshold * trace_run[i])):
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



def rift_cataloger(atl06_data,verbose=True):
    '''
    INPUT: atl06_data, a pandas DataFrame with each "row" being the data from a single ATL06 file, 
            masked to the Antarctic Ice Shelves. The keys of the dictionary are defined in arc.ingest().
            
    OUTPUT: rift_obs, a dictionary of lists, with each "row" corresponding to a single rift observation.  
            The Column headings are:
            
            d-start       - Along-track distance to the landward rift wall (d = sqrt(x**2 + y**2))
            d-end         - Along-track distance to the seaward rift wall
            x-start       - Landward rift wall x location in polar stereographic coordinates (EPSG: 3031) (1)
            y-start       - Landward rift wall y location in polar stereographic coordinates (1)
            x-end         - Seaward rift wall x location in polar stereographic coordinates (1)
            y-end         - Seaward rift wall y location in polar stereographic coordinates (1)
            x-centroid    - Rift mid-point x location in polar stereographic coordinates 
            y-centroid    - Rift mid-point y location in polar stereographic coordinates
            lat-centroid  - Rift mid-point latitude
            long-centroid - Rift mid-point longitude
            width         - Rift width in ICESat-2 ground track geometry (distance between d-end and d-start) (2)
            d_seaward     - Along-track distance to location of seaward height measurement for landward-seaward offset (3)
            d_landward    - Along-track distance to location of landward height measurement for landward-seaward offset (3)
            h_seaward     - height at d_seaward for landward-seaward offset (3) ### IN DEVELOPMENT ###
            h_landward    - height at d_landward for landward-seaward offset (3) ### IN DEVELOPMENT ###
            sl_offset     - landward-seaward height offset (3) ### IN DEVELOPMENT ###
            time          - from ATL06 input data - Time (YYYY-MM-DD hh:mm:ss.ssssss)
            rgt           - from ATL06 input data - Reference Ground Track
            azimuth       - from ATL06 input data - mean of beam azimuth between d-start and d-end
            sigma         - from ATL06 input data - mean of sigma between d-start and d-end
            h             - mean freeboard between d-start and d-end (3) ### IN DEVELOPMENT for melange thickness ###
            beam          - from ATL06 input data - beam
            data_row      - Counter of main loop (i). Provides a quick way for linking rifts from same cycle, RGT and beam
            confidence    - Rift measurement confidence (high/medium/low)
            
            (1) In general these are the landward and seaward rift walls, but they may be reversed in some areas.
            (2) The rift width in ICESat-2 ground track geometry is later corrected to an estimate of actual rift width
                using the angular offset between the RGT and the rift-perpendicular axis.
            (3) Landward-seaward offset and melange thickness code in development
            
    '''
    rift_obs = {
        "d-start": [],
        "d-end": [],
        "x-start": [],
        "y-start": [],
        "x-end": [],
        "y-end": [],
        "x-centroid": [],
        "y-centroid": [],
        "lat-centroid": [],
        "lon-centroid": [],
        "width": [],
        "d_seaward": [],
        "d_landward": [],
        "h_seaward": [],
        "h_landward": [],
        "sl_offset": [],
        "time": [],
        "rgt": [],
        "azimuth": [],
        "sigma": [],
        "h":[],
        "beam":[],
        "data_row":[],
        "confidence":[]
    } 
    
    ttstart = t.perf_counter()
    
    for i, row in atl06_data.iterrows():

        qual    = row["quality"]
        ht      = row["h"]
        geoid   = row["geoid"]
        tides   = row["tides"]
        mdt     = -1.1
        
        if len(qual) == 0:
            continue
            
        if len(ht) < 10:
            continue

        h = ht - geoid - tides - mdt
            
        # Only allow a certain percentage of data to be flagged as low quality
        percent_low_quality = sum( qual==1 )  / len(qual)
        if percent_low_quality > 0.2:
            # if there is a high percentage of low quality data
            # try removing bad segments of the pass
            # if that doesn't work, skip
            h_data = np.ma.getdata(h)
            h_mask = np.ma.getmask(h)
            h_new_mask = []
            original = 0
            for tl in range(0,len(h),200):
                subqual = qual[tl:tl+200]
                submask = h_mask[tl:tl+200]
                sub_percent_low_qual = sum(subqual) / len(subqual)
                if sub_percent_low_qual > 0.2:
                    h_new_mask[tl:tl+200] = np.ones(len(submask), dtype=bool)
                else:
                    h_new_mask[tl:tl+200] = submask
                    original = 1
                    
            h = np.ma.array(h_data,mask=h_new_mask)
            if original != 1:
                continue
            
        # Data product is posted at 20m
        # allow the spacing to be slightly greater (25m)
        # and ensure most gaps (e.g. 97%) are less than this
        spacing = (max(np.sqrt(row['x']**2 + row['y']**2)) - min(np.sqrt(row['x']**2 + row['y']**2))) \
        / len(np.sqrt(row['x']**2 + row['y']**2))
        
        pass_x = row["x"]
        pass_y = row["y"]
        pass_d = np.sqrt(pass_x**2 + pass_y**2)
        
        pass_sep = np.zeros(len(pass_d)-1)
        pass_below = np.zeros(len(pass_d)-1)
                
        for sep in range(1,len(pass_d)):
            pass_sep[sep-1]=pass_d[sep]-pass_d[sep-1]
            if abs(pass_d[sep]-pass_d[sep-1]) < 25:
                pass_below[sep-1] = 1   
        pass_percent = (np.sum(pass_below)/len(pass_below))*100
        
        if (spacing > 25) & (pass_percent < 97):
            continue
        
        #-----------------------------
        # Define some thresholds
        bad_threshold = 100           # Gross error threshold (m)
                                      # points with an elevation greater than plus bad threshold
                                      # and less than minus bad threshold are discarded
                                      # This will have to change for different ice shelves
        run_mean_dist = 10000         # distance in m each side of the point
                                      # to smooth over
        wall_threshold_low = 0.5      # fraction of running mean height (>0, <1) that the subset
        wall_threshold_hi = 0.8       # around the rift must reach, either side of the lowest point
                                      # for it to be considered that rift walls have been found
                                      # <0.5 = low (default to detection width) >0.5 = medium, >0.8 = high
        rift_qual_threshold = 0.25    # the fraction (>0, <1) of points within the rift
                                      # that can have a low quality flag from ATL06 input
        wall_mean_sep_threshold = 50  # maximum spacing of wall points
        dist_half_mini = 100          # distance in m each side of a point on the walls to conduct
                                      # linear regression over to find steepest point
        lr_threshold = 5              # minimum number of points for linear regression
        sl_dist_threshold = 100       # maximum distance in m the points for seaward-landward offset
                                      # can be from the rift walls. Otherwise return 'nan'
        
        transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
                                      #transform polar stereographic to lat/lon
        
        
        #-----------------------------
        # load necessary arrays
        x       = row["x"]
        y       = row["y"]
        d       = np.sqrt(x**2 + y**2) #distance in m
        rgt     = row["rgt"]
        beam    = row["beam"]
        time    = row["time"]
        azimuth = row["azimuth"]
        sigma   = row["h_sig"]
        
        #-----------------------------
        # return some information
        if verbose == True:
            print("")
            print("processing:")
            print("rgt : "+str(rgt))
            print("beam: "+beam)
            print("date: "+str(time))
            
        #-----------------------------
        # flip arrays if necessary
        if d[len(d)-1] < d[0]:
            x       = np.flip(x)
            y       = np.flip(y)
            d       = np.flip(d)
            h       = np.flip(h)
            ht      = np.flip(ht)
            geoid   = np.flip(geoid)
            qual    = np.flip(qual)
            azimuth = np.flip(azimuth)
            sigma   = np.flip(sigma)
            
            if verbose == True:
                print("descending pass - flipping arrays")
        else:
            if verbose == True:
                print("ascending pass")
                
        #-----------------------------
        # make filter for bad data caused by clouds in ht 
        # and gross errors in ht and geoid using "bad_threshold"
        new_qual = np.where((ht>bad_threshold) | (ht<-bad_threshold) | (geoid>bad_threshold) | (geoid<-bad_threshold), 1, 0)
        
        #-----------------------------
        # apply the filter to x, y, d, h, original quality, sigma and azimuth
        d = d[new_qual==0]
        x = x[new_qual==0]
        y = y[new_qual==0]
        h = h[new_qual==0]
        qual = qual[new_qual==0]
        azimuth = azimuth[new_qual==0]
        sigma = sigma[new_qual==0]
        
        if len(h) < 10: #skip if following filtering there are <10 points
            continue
        
        #-----------------------------
        # calculate the running mean (smoothed surface)
        
        spacing = (max(d) - min(d)) / len(d)
        run_mean_nr = round(run_mean_dist / spacing)
        h_run = pd.DataFrame(h).rolling((2*run_mean_nr)+1, min_periods=1, center=True).mean()
        h_run = h_run.values.flatten()
        
        # the area to search for a rift wall is later defined as where height >= running mean height
        # (because rift depresses running mean height, this can generally be found
        # but not for rifts at the start or end of the trace
        # tying the running mean start and end to the height ensures walls can always be found for 
        # rifts near grounding line and calving front
        # This can be altered when the detection algorithm is updated
        h_run_tied = pd.DataFrame(h).rolling((2*run_mean_nr)+1, min_periods=1, center=True).mean()
        h_run_tied = h_run_tied.values.flatten()
        h_run_tied[0] = -99
        h_run_tied[len(h_run_tied)-1] = -99
    
        #-----------------------------
        # pass trace to rift detector
        # INPUTS (mandatory): h, h_run, d (surface height, running mean/smoothed surface height, distance)
        # INPUTS (optional): threshold (fraction of running mean >0, <1, default = 0.6)
        # OUTPUTS: A series of start and stop indices for "rift" detections
               
        rift_list = rift_detector(h,h_run,d,0.5)
        
        nr_rifts = len(rift_list)   
        
        #loop through the rifts detected by the (temporary) rift detector
        if len(rift_list) > 0:
            if verbose == True:
                print("rift detector found "+str(nr_rifts)+" possible rifts")
                
            #-----------------------------
            # loop through the rift detections
            # filter out low confidence detections
            # measure width
            
            for rift in rift_list:
                if verbose == True:
                    print("------------------------------")
                    print("rift: "+str(rift))
                
                # extract start and end indices of rift detection
                rift_start = rift[0]
                rift_end   = rift[1]
                
                # ignore if first or last h point is in the possible rift
                # i.e. grounding line rift or calving front
                # because we won't find another wall
                if rift_start != 0:
                    if rift_end != len(h)-1:
                        
                        # convert to distance
                        dist_start = d[rift_start]
                        dist_end   = d[rift_end]
                        dist_rift  = dist_end - dist_start
                        dist_start_x = x[rift_start]
                        dist_end_x = x[rift_end]
                        dist_start_y = y[rift_start]
                        dist_end_y = y[rift_end]
                        
                        #-----------------------------
                        # check the detected rift isn't completely within a masked region

                        if (len(h[rift_start:rift_end])>sum(np.ma.getmask(h[rift_start:rift_end]).astype(int))) & \
                        (len(h[rift_start:rift_end])>sum(qual[rift_start:rift_end])):

                            #-----------------------------
                            # find the lowest point in the rift
                            rift_low_idx = int(np.where(h[rift_start:rift_end] == h[rift_start:rift_end].min()) + rift_start)
                            height_low   = h[rift_low_idx]
                            dist_low     = d[rift_low_idx]
                            
                            #-----------------------------
                            # work outwards from the lowest point
                            # define subsets of the surface, smoothed surface and distance arrays
                            # check there is something within 2 * rift detection width
                            # either side of lowest point that can be considered a rift wall
                            idx_around_start = next(x for x, val in enumerate(d) if val > dist_low - (2*dist_rift))
                            idx_around_end = len(d)-1 #if search window is beyond the end of the trace
                            try:
                                idx_around_end = next(x for x, val in enumerate(d) if val > dist_low + (2*dist_rift))-1
                            except:
                                if verbose==True:
                                    print("rift near end of trace, rift end area defaulting to end of trace") 
                                
                            h_sub = h[idx_around_start:idx_around_end]
                            h_run_sub = h_run[idx_around_start:idx_around_end]

                            # index of rift lowest point within the subset
                            sub_rift_low_idx = np.where(h_sub==height_low)
                            sub_rift_low_idx = np.array(sub_rift_low_idx).flatten()

                            #-----------------------------
                            # ensure that there are walls within this subset around the rift
                            # search for points above threshold (default = 0.8) of running mean
                            sub_walls_low = np.where(h_sub > (wall_threshold_low * h_run_sub))
                            sub_walls_hi  = np.where(h_sub > (wall_threshold_hi * h_run_sub))

                            #-----------------------------
                            # ensure the proportion of low quality data is low
                            qual_walls = qual[idx_around_start:idx_around_end]
                            percent_low_qual_walls = sum(qual_walls==1) / len(qual_walls)

                            if(len(sub_walls_low[0]) > 0) & (percent_low_qual_walls < rift_qual_threshold):
                                #ie something was found above the threshold
                                #and only a small number of rift points are low quality
                                
                                # work out whether there are points on either side of the rift that exceed
                                # the thresholds to be considered medium or high confidence detections
                                if(np.amin(sub_walls_low) < sub_rift_low_idx) & (np.amax(sub_walls_low) > sub_rift_low_idx):
                                    confidence = "medium"
                                    if(len(sub_walls_hi[0]) > 0):
                                        if(np.amin(sub_walls_hi) < sub_rift_low_idx) &\
                                        (np.amax(sub_walls_hi) > sub_rift_low_idx):
                                            confidence = "high"
                                
                                    if verbose == True:
                                        print("processing rift: both walls found within distance limit")
                                    
                                    #-----------------------------
                                    # now define a new subset encompassing the area around the lowest point
                                    # where points exceed the running mean

                                    #walls = np.where(h >= h_run)
                                    walls = np.where(h >= h_run_tied)
                                    walls = np.array(walls).flatten()
                                    
                                    rift_start_idx = walls[walls.searchsorted(rift_low_idx,'right')-1]
                                    rift_end_idx = walls[walls.searchsorted(rift_low_idx,'right')]
                                    
                                    #-----------------------------
                                    # extract arrays for rift start to lowest point (where the landward wall will be)
                                    # and lowest point to rift end (where the seaward wall will be)
                                    h_sub_walls_before = h[rift_start_idx:rift_low_idx+1]
                                    d_sub_walls_before = d[rift_start_idx:rift_low_idx+1]
                                    x_sub_walls_before = x[rift_start_idx:rift_low_idx+1]
                                    y_sub_walls_before = y[rift_start_idx:rift_low_idx+1]
                                    mask_before = np.ma.getmask(h_sub_walls_before)
                                    
                                    h_sub_walls_after = h[rift_low_idx:rift_end_idx+1]
                                    d_sub_walls_after = d[rift_low_idx:rift_end_idx+1]
                                    x_sub_walls_after = x[rift_low_idx:rift_end_idx+1]
                                    y_sub_walls_after = y[rift_low_idx:rift_end_idx+1]
                                    mask_after = np.ma.getmask(h_sub_walls_after)

                                    # threshold of half rift depth for finding walls
                                    updown_threshold = 0.5*(((h_run[rift_start_idx:rift_end_idx]).mean())-height_low)
                                    
                                    h_sub_walls_before = h_sub_walls_before[mask_before==False]
                                    d_sub_walls_before = d_sub_walls_before[mask_before==False]
                                    x_sub_walls_before = x_sub_walls_before[mask_before==False]
                                    y_sub_walls_before = y_sub_walls_before[mask_before==False]
                                    h_sub_walls_after  = h_sub_walls_after[mask_after==False]
                                    d_sub_walls_after  = d_sub_walls_after[mask_after==False]
                                    x_sub_walls_after  = x_sub_walls_after[mask_after==False]
                                    y_sub_walls_after  = y_sub_walls_after[mask_after==False]
                                    
                                    h_sub_walls_before_int = list(range(0,len(h_sub_walls_before)))
                                    h_sub_walls_after_int = list(range(0,len(h_sub_walls_after)))
                                    
                                    #mean separation of points on the walls
                                    d_sub_walls_before_mean_sep =\
                                    (np.max(d_sub_walls_before) - np.min(d_sub_walls_before)) / (len(d_sub_walls_before)-1)
                                    
                                    d_sub_walls_after_mean_sep =\
                                    (np.max(d_sub_walls_after) - np.min(d_sub_walls_after)) / (len(d_sub_walls_after)-1)
                                    
                                    if ((np.max(h_sub_walls_before)-height_low > updown_threshold) &\
                                        (np.max(h_sub_walls_after)-height_low > updown_threshold) &\
                                        (d_sub_walls_before_mean_sep < wall_mean_sep_threshold) &
                                        (d_sub_walls_after_mean_sep < wall_mean_sep_threshold)):
                                        
                                        #-----------------------------
                                        # make sure there are a sufficient number of points in the walls
                                        # to calculate steepest slope location
                                        if (len(h_sub_walls_before) > 3)  & (len(h_sub_walls_after) > 3):
                                            #----------
                                            # DETECTION OF RIFT WALLS
                                            # find sections of the before array that are continuously going down

                                            # initialise
                                            down = np.zeros(len(h_sub_walls_before)-1)
                                            down_h_diff = []

                                            # find adjacent heights that are descending (1)
                                            for b in range (0,len(down)):
                                                b1 = h_sub_walls_before[b+1]
                                                b2 = h_sub_walls_before[b]
                                                if b1 < b2:
                                                    down[b] = 1

                                            down_regions = scipy.ndimage.find_objects(scipy.ndimage.label(down)[0])

                                            # for each section that is continuously going down
                                            # calculate the height difference
                                            for down_region in down_regions:
                                                down_region = down_region[0]
                                                down_h_start_idx = h_sub_walls_before_int[down_region][0]
                                                down_h_end_temp = h_sub_walls_before_int[down_region][-1]
                                                down_h_end_idx = h_sub_walls_before_int[down_h_end_temp]+2
                                                down_h = h_sub_walls_before[down_h_start_idx:down_h_end_idx]
                                                down_h_diff.append(down_h[0] - down_h[len(down_h)-1])

                                            # does anything exceed half of rift depth?
                                            # if so, the closest to the lowest point is probably the wall
                                            # else the wall is the section with the largest h decrease
                                            down_above_threshold = (down_h_diff > updown_threshold).astype(int)

                                            if sum(down_above_threshold) > 0:
                                                down_h_max_idx = max(np.where(down_above_threshold==1)[0])
                                            else:
                                                down_h_max_idx = np.where(down_h_diff==max(down_h_diff))
                                                down_h_max_idx = int((down_h_max_idx[0]).flatten())

                                            down_selected_start_idx = h_sub_walls_before_int[down_regions[down_h_max_idx][0]][0]
                                            down_selected_end_temp = h_sub_walls_before_int[down_regions[down_h_max_idx][0]][-1]
                                            down_selected_end_idx = h_sub_walls_before_int[down_selected_end_temp] + 2

                                            # subset arrays to the section identified as the rift wall
                                            # use these as the centers for the mini linear regression
                                            h_array_downslope = h_sub_walls_before[down_selected_start_idx:down_selected_end_idx]
                                            d_array_downslope = d_sub_walls_before[down_selected_start_idx:down_selected_end_idx]
                                            x_array_downslope = x_sub_walls_before[down_selected_start_idx:down_selected_end_idx]
                                            y_array_downslope = y_sub_walls_before[down_selected_start_idx:down_selected_end_idx]

                                            #-----------------------------
                                            # find sections of the array array that are continuously going up

                                            # initialise
                                            up = np.zeros(len(h_sub_walls_after)-1)
                                            up_h_diff = []

                                            # find adjacent heights that are ascending (1)
                                            for a in range (0,len(up)):
                                                a1 = h_sub_walls_after[a]
                                                a2 = h_sub_walls_after[a+1]
                                                if a2 > a1:
                                                    up[a] = 1

                                            up_regions = scipy.ndimage.find_objects(scipy.ndimage.label(up)[0])

                                            # for each section that is continuously going up
                                            # calculate the height difference
                                            for up_region in up_regions:
                                                up_region = up_region[0]
                                                up_h_start_idx = h_sub_walls_after_int[up_region][0]
                                                up_h_end_temp = h_sub_walls_after_int[up_region][-1]
                                                up_h_end_idx = h_sub_walls_after_int[up_h_end_temp]+2
                                                up_h = h_sub_walls_after[up_h_start_idx:up_h_end_idx]
                                                up_h_diff.append(up_h[len(up_h)-1] - up_h[0])

                                            # does anything exceed half of rift depth?
                                            # if so, the closest to the lowest point it probably the wall
                                            # else the wall is the section with the largest h increse
                                            up_above_threshold = (up_h_diff > updown_threshold).astype(int)

                                            if sum(up_above_threshold) > 0:
                                                up_h_max_idx = min(np.where(up_above_threshold==1)[0])
                                            else:
                                                up_h_max_idx = np.where(up_h_diff==max(up_h_diff))
                                                up_h_max_idx = int((up_h_max_idx[0]).flatten())

                                            up_selected_start_idx = h_sub_walls_after_int[up_regions[up_h_max_idx][0]][0]
                                            up_selected_end_temp = h_sub_walls_after_int[up_regions[up_h_max_idx][0]][-1]
                                            up_selected_end_idx = h_sub_walls_after_int[up_selected_end_temp] + 2

                                            # subset arrays to the section identified as the rift wall
                                            # use these as the centers for the mini linear regression
                                            h_array_upslope = h_sub_walls_after[up_selected_start_idx:up_selected_end_idx]
                                            d_array_upslope = d_sub_walls_after[up_selected_start_idx:up_selected_end_idx]
                                            x_array_upslope = x_sub_walls_after[up_selected_start_idx:up_selected_end_idx]
                                            y_array_upslope = y_sub_walls_after[up_selected_start_idx:up_selected_end_idx]

                                            #-----------------------------
                                            # SELECTION OF STEEPEST SLOPES
                                            # perform mini linear regression over small distances to find
                                            # the steepest slopes of the array subsets defined as the rift walls

                                            # initialise
                                            d_lr_down = []
                                            x_lr_down = []
                                            y_lr_down = []
                                            slope_lr_down = []
                                            d_lr_up = []
                                            x_lr_up = []
                                            y_lr_up = []
                                            slope_lr_up = []


                                            # loop through each segment on the rift wall
                                            # perform linear regression using all points within the distance limit
                                            # down
                                            for dr in range (0,len(d_array_downslope)):
                                                dist_min = d_array_downslope[dr] - dist_half_mini
                                                dist_max = d_array_downslope[dr] + dist_half_mini
                                                idx_points = np.where((d > dist_min) & (d < dist_max))
                                                d_lr_down_points = d[idx_points]
                                                x_lr_down_points = x[idx_points]
                                                y_lr_down_points = y[idx_points]
                                                h_lr_down_points = h[idx_points]
                                                nr_points = np.ma.MaskedArray.count(h_lr_down_points)
                                                if nr_points > lr_threshold:
                                                    d_lr_down.append(np.mean(d_lr_down_points))
                                                    x_lr_down.append(np.mean(x_lr_down_points))
                                                    y_lr_down.append(np.mean(y_lr_down_points))
                                                    lr_slope = linregress(d_lr_down_points,h_lr_down_points).slope
                                                    slope_lr_down.append(lr_slope)

                                            # up
                                            for ur in range (0,len(d_array_upslope)):
                                                dist_min = d_array_upslope[ur] - dist_half_mini
                                                dist_max = d_array_upslope[ur] + dist_half_mini
                                                idx_points = np.where((d > dist_min) & (d < dist_max))
                                                d_lr_up_points = d[idx_points]
                                                x_lr_up_points = x[idx_points]
                                                y_lr_up_points = y[idx_points]
                                                h_lr_up_points = h[idx_points]
                                                nr_points = np.ma.MaskedArray.count(h_lr_up_points)
                                                if nr_points > lr_threshold:
                                                    d_lr_up.append(np.mean(d_lr_up_points))
                                                    x_lr_up.append(np.mean(x_lr_up_points))
                                                    y_lr_up.append(np.mean(y_lr_up_points))
                                                    lr_slope = linregress(d_lr_up_points,h_lr_up_points).slope
                                                    slope_lr_up.append(lr_slope)


                                            #-----------------------------
                                            # find the steepest parts of the two rift walls
                                            if (len(slope_lr_down) > 0) & (len(slope_lr_up) > 0):
                                                slope_down_idx = np.argmin(slope_lr_down)
                                                slope_up_idx   = np.argmax(slope_lr_up)
                                                slope_down_d   = d_lr_down[slope_down_idx]
                                                slope_up_d     = d_lr_up[slope_up_idx]
                                                slope_down_x   = x_lr_down[slope_down_idx]
                                                slope_up_x     = x_lr_up[slope_up_idx]
                                                slope_down_y   = y_lr_down[slope_down_idx]
                                                slope_up_y     = y_lr_up[slope_up_idx]

                                                # calculate the coordinates of the center of the rift (polar stereographic)
                                                rift_centroid_x = (slope_down_x + slope_up_x) / 2
                                                rift_centroid_y = (slope_down_y + slope_up_y) / 2

                                                # calculate the coordinates of the center of the rift (lat/lon)
                                                [rift_centroid_lat,rift_centroid_lon] = \
                                                transformer.transform(rift_centroid_x,rift_centroid_y)

                                                # calculate rift width
                                                rift_width = slope_up_d - slope_down_d
                                                if verbose == True:
                                                    print("rift width (satellite geometry): "+str(round(rift_width,2))+" m")

                                                if rift_width > 40: #platelet size

                                                    #-----------------------------
                                                    # calculate the mean azimuth and h_sig of the rift

                                                    # first have to find the indices of the nearest points
                                                    # (because the mini liner regression values are assigned to the
                                                    # mean of the points in each calculation, which will not line up
                                                    # with input data)
                                                    az_start_idx = np.searchsorted(d,slope_down_d,side="left")
                                                    az_end_idx = np.searchsorted(d,slope_up_d,side="left")-1
                                                    rift_azimuth = azimuth[az_start_idx:az_end_idx].mean()
                                                    rift_sigma = sigma[az_start_idx:az_end_idx].mean()
                                                    rift_h = h[az_start_idx:az_end_idx].mean()
                                                    
                                                    #-----------------------------
                                                    # extract values and calculate landward-seaward offset
                                                    # Catherine suggested using the h values from immediately 
                                                    # outside the rift detection
                                                    
                                                    # landward slope is 'slope_down_d' (down into rift)
                                                    # seaward slope is 'slope_up_d' (up out of rift)
                                                    # arrays are 'd' and 'h'
                                                    
                                                    landward_idx = max(np.where(d<slope_down_d)[0]) - 1
                                                    seaward_idx = min(np.where(d>slope_up_d)[0]) + 1

                                                    landward_d = d[landward_idx]
                                                    seaward_d = d[seaward_idx]

                                                    landward_dist = abs(landward_d - slope_down_d)
                                                    seaward_dist = abs(seaward_d - slope_up_d)

                                                    if (landward_dist < sl_dist_threshold) &\
                                                    (seaward_dist < sl_dist_threshold):
                                                        landward_h = h[landward_idx]
                                                        seaward_h = h[seaward_idx]
                                                        sl_offset = seaward_h - landward_h
                                                    else:
                                                        landward_h = np.nan
                                                        seaward_h = np.nan
                                                        sl_offset = np.nan
                                                        landward_d = np.nan
                                                        seaward_d = np.nan

                                                    #-----------------------------
                                                    # append to list for output
                                                    rc = len(rift_obs["d-start"]) #rift_counter

                                                    # if this rift overlaps with previous rift
                                                    # use outermost detected walls of the two rifts
                                                    # and recalculate other values
                                                    # then add to the output
                                                    if rc > 0:
                                                        # if the beginning of the current rift
                                                        # is inside the bounds of the previous rift
                                                        if (i == rift_obs["data_row"][rc-1]) & \
                                                        (slope_down_d < rift_obs["d-end"][rc-1]):

                                                            if rift_obs["d-start"][rc-1] > slope_down_d:
                                                                rift_obs["d-start"][rc-1] = slope_down_d
                                                                rift_obs["x-start"][rc-1] = slope_down_x
                                                                rift_obs["y-start"][rc-1] = slope_down_y
                                                                rift_obs["h_landward"][rc-1] = landward_h
                                                                rift_obs["d_landward"][rc-1] = landward_d
                                                        

                                                            if rift_obs["d-end"][rc-1] < slope_up_d:
                                                                rift_obs["d-end"][rc-1] = slope_up_d
                                                                rift_obs["x-end"][rc-1] = slope_up_x
                                                                rift_obs["y-end"][rc-1] = slope_up_y
                                                                rift_obs["h_seaward"][rc-1] = seaward_h
                                                                rift_obs["d_seaward"][rc-1] = seaward_d

                                                            rift_obs["sl_offset"][rc-1] = \
                                                            rift_obs["h_seaward"][rc-1] - rift_obs["h_landward"][rc-1]
                                                            
                                                                
                                                            rift_obs["x-centroid"][rc-1] = \
                                                            (rift_obs["x-start"][rc-1] + rift_obs["x-end"][rc-1]) / 2
                                                            rift_obs["y-centroid"][rc-1] = \
                                                            (rift_obs["y-start"][rc-1] + rift_obs["y-end"][rc-1]) / 2

                                                            [rift_centroid_lat,rift_centroid_lon] = \
                                                            transformer.transform(rift_obs["x-centroid"][rc-1], \
                                                                                  rift_obs["y-centroid"][rc-1])

                                                            rift_obs["lat-centroid"][rc-1] = rift_centroid_lat
                                                            rift_obs["lon-centroid"][rc-1] = rift_centroid_lon

                                                            rift_obs["width"][rc-1] = \
                                                            rift_obs["d-end"][rc-1] - rift_obs["d-start"][rc-1]

                                                            az_start_idx =\
                                                            np.searchsorted(d,rift_obs["d-start"][rc-1],side="left")
                                                            
                                                            az_end_idx =\
                                                            np.searchsorted(d,rift_obs["d-end"][rc-1],side="left")-1

                                                            rift_azimuth = azimuth[az_start_idx:az_end_idx].mean()
                                                            rift_sigma = sigma[az_start_idx:az_end_idx].mean()
                                                            rift_h = h[az_start_idx:az_end_idx].mean()

                                                            rift_obs["azimuth"][rc-1] = rift_azimuth
                                                            rift_obs["sigma"][rc-1] = rift_sigma
                                                            rift_obs["h"][rc-1] = rift_h
                                                            
                                                            rift_obs["confidence"][rc-1] = confidence
                                                            

                                                        else:
                                                            # if this rift doesn't overlap with the previous rift
                                                            # then add to the output
                                                            rift_obs["d-start"].append(slope_down_d)
                                                            rift_obs["d-end"].append(slope_up_d)
                                                            rift_obs["x-start"].append(slope_down_x)
                                                            rift_obs["y-start"].append(slope_down_y)
                                                            rift_obs["x-end"].append(slope_up_x)
                                                            rift_obs["y-end"].append(slope_up_y)
                                                            rift_obs["x-centroid"].append(rift_centroid_x)
                                                            rift_obs["y-centroid"].append(rift_centroid_y)
                                                            rift_obs["lat-centroid"].append(rift_centroid_lat)
                                                            rift_obs["lon-centroid"].append(rift_centroid_lon)
                                                            rift_obs["width"].append(rift_width)
                                                            rift_obs["d_seaward"].append(seaward_d)
                                                            rift_obs["d_landward"].append(landward_d)
                                                            rift_obs["h_seaward"].append(seaward_h)
                                                            rift_obs["h_landward"].append(landward_h)
                                                            rift_obs["sl_offset"].append(sl_offset)
                                                            rift_obs["time"].append(time)
                                                            rift_obs["rgt"].append(rgt)
                                                            rift_obs["azimuth"].append(rift_azimuth)
                                                            rift_obs["sigma"].append(rift_sigma)
                                                            rift_obs["h"].append(rift_h)
                                                            rift_obs["beam"].append(beam)
                                                            rift_obs["data_row"].append(i)
                                                            rift_obs["confidence"].append(confidence)

                                                    else:
                                                        # if this is the first rift
                                                        # add to the ooutput
                                                        rift_obs["d-start"].append(slope_down_d)
                                                        rift_obs["d-end"].append(slope_up_d)
                                                        rift_obs["x-start"].append(slope_down_x)
                                                        rift_obs["y-start"].append(slope_down_y)
                                                        rift_obs["x-end"].append(slope_up_x)
                                                        rift_obs["y-end"].append(slope_up_y)
                                                        rift_obs["x-centroid"].append(rift_centroid_x)
                                                        rift_obs["y-centroid"].append(rift_centroid_y)
                                                        rift_obs["lat-centroid"].append(rift_centroid_lat)
                                                        rift_obs["lon-centroid"].append(rift_centroid_lon)
                                                        rift_obs["width"].append(rift_width)
                                                        rift_obs["d_seaward"].append(seaward_d)
                                                        rift_obs["d_landward"].append(landward_d)
                                                        rift_obs["h_seaward"].append(seaward_h)
                                                        rift_obs["h_landward"].append(landward_h)
                                                        rift_obs["sl_offset"].append(sl_offset)
                                                        rift_obs["time"].append(time)
                                                        rift_obs["rgt"].append(rgt)
                                                        rift_obs["azimuth"].append(rift_azimuth)
                                                        rift_obs["sigma"].append(rift_sigma)
                                                        rift_obs["h"].append(rift_h)
                                                        rift_obs["beam"].append(beam)
                                                        rift_obs["data_row"].append(i)
                                                        rift_obs["confidence"].append(confidence)
                                                        
                                                        

                                                else:
                                                    # rift width <40m, i.e. the size of an ATL06 platelet
                                                    # skip the rift
                                                    if verbose == True:
                                                        print("skipping rift: rift width below platelet size")
                                                        print("suggests wall finding error")

                                            else:
                                                # couldn't calculate the the steepest slope on one or both walls
                                                # within the quality thresholds
                                                # then we default to the initial rift width from the are below
                                                # 50% (or chosen threshold) of running mean
                                                # ----------
                                                # in the Halloween Crack case study, this was a small number
                                                # of narrow rift areas (~100m) where an iceberg or semi-detached block
                                                # bisected the rift.
                                                # Low confidence detections were discarded unless they are merged with
                                                # a high confidence detection. However, keeping them helps improve the
                                                # accuracy of those measurements which are bisected.
                                                # ----------
                                                
                                                confidence = "low"
                                    
                                                # calculate the coordinates of the center of the rift (polar stereographic)
                                                rift_centroid_x = (dist_start_x + dist_end_x) / 2
                                                rift_centroid_y = (dist_start_y + dist_end_y) / 2

                                                # calculate the coordinates of the center of the rift (lat/lon)
                                                [rift_centroid_lat,rift_centroid_lon] = \
                                                transformer.transform(rift_centroid_x,rift_centroid_y)

                                                # calculate rift width
                                                rift_width = dist_end - dist_start

                                                seaward_d = np.nan
                                                landward_d = np.nan
                                                seaward_h = np.nan
                                                landward_h = np.nan
                                                sl_offset = np.nan

                                                rift_azimuth = azimuth[rift_start:rift_end+1].mean()
                                                rift_sigma = sigma[rift_start:rift_end+1].mean()
                                                rift_h = np.nan
                                    
                                                rift_obs["d-start"].append(dist_start)
                                                rift_obs["d-end"].append(dist_end)
                                                rift_obs["x-start"].append(dist_start_x)
                                                rift_obs["y-start"].append(dist_start_y)
                                                rift_obs["x-end"].append(dist_end_x)
                                                rift_obs["y-end"].append(dist_end_y)
                                                rift_obs["x-centroid"].append(rift_centroid_x)
                                                rift_obs["y-centroid"].append(rift_centroid_y)
                                                rift_obs["lat-centroid"].append(rift_centroid_lat)
                                                rift_obs["lon-centroid"].append(rift_centroid_lon)
                                                rift_obs["width"].append(rift_width)
                                                rift_obs["d_seaward"].append(seaward_d)
                                                rift_obs["d_landward"].append(landward_d)
                                                rift_obs["h_seaward"].append(seaward_h)
                                                rift_obs["h_landward"].append(landward_h)
                                                rift_obs["sl_offset"].append(sl_offset)
                                                rift_obs["time"].append(time)
                                                rift_obs["rgt"].append(rgt)
                                                rift_obs["azimuth"].append(rift_azimuth)
                                                rift_obs["sigma"].append(rift_sigma)
                                                rift_obs["h"].append(rift_h)
                                                rift_obs["beam"].append(beam)
                                                rift_obs["data_row"].append(i)
                                                rift_obs["confidence"].append(confidence) 
                                                
                                                
                                                
                                                if verbose == True:
                                                    #print("skipping rift: insufficient points in the walls")
                                                    print("cannot refine rift measurements")
                                                    print("defaulting to initial estimate")
                                                    print("confidence: low")

                                        else:
                                            # couldn't calculate the the steepest slope on one or both walls
                                            # because there were too few quality ATL06 measurements
                                            # then we default to the initial rift width from the are below
                                            # 50% (or chosen threshold) of running mean
                                            # ----------
                                            # see Halloween Crack case study information above
                                            # ----------
                                            
                                            confidence = "low"
                                            
                                            # calculate the coordinates of the center of the rift (polar stereographic)
                                            rift_centroid_x = (dist_start_x + dist_end_x) / 2
                                            rift_centroid_y = (dist_start_y + dist_end_y) / 2

                                            # calculate the coordinates of the center of the rift (lat/lon)
                                            [rift_centroid_lat,rift_centroid_lon] = \
                                            transformer.transform(rift_centroid_x,rift_centroid_y)

                                            # calculate rift width
                                            rift_width = dist_end - dist_start

                                            seaward_d = np.nan
                                            landward_d = np.nan
                                            seaward_h = np.nan
                                            landward_h = np.nan
                                            sl_offset = np.nan

                                            rift_azimuth = azimuth[rift_start:rift_end+1].mean()
                                            rift_sigma = sigma[rift_start:rift_end+1].mean()
                                            rift_h = np.nan
                                    
                                            rift_obs["d-start"].append(dist_start)
                                            rift_obs["d-end"].append(dist_end)
                                            rift_obs["x-start"].append(dist_start_x)
                                            rift_obs["y-start"].append(dist_start_y)
                                            rift_obs["x-end"].append(dist_end_x)
                                            rift_obs["y-end"].append(dist_end_y)
                                            rift_obs["x-centroid"].append(rift_centroid_x)
                                            rift_obs["y-centroid"].append(rift_centroid_y)
                                            rift_obs["lat-centroid"].append(rift_centroid_lat)
                                            rift_obs["lon-centroid"].append(rift_centroid_lon)
                                            rift_obs["width"].append(rift_width)
                                            rift_obs["d_seaward"].append(seaward_d)
                                            rift_obs["d_landward"].append(landward_d)
                                            rift_obs["h_seaward"].append(seaward_h)
                                            rift_obs["h_landward"].append(landward_h)
                                            rift_obs["sl_offset"].append(sl_offset)
                                            rift_obs["time"].append(time)
                                            rift_obs["rgt"].append(rgt)
                                            rift_obs["azimuth"].append(rift_azimuth)
                                            rift_obs["sigma"].append(rift_sigma)
                                            rift_obs["h"].append(rift_h)
                                            rift_obs["beam"].append(beam)
                                            rift_obs["data_row"].append(i)
                                            rift_obs["confidence"].append(confidence) 
                                            
                                            if verbose == True:
                                                print("cannot refine rift measurements")
                                                print("defaulting to initial estimate")
                                                print("confidence: low")
                                            
                                    else: # points within reasonable distance meet wall threshold only on one side
                                        if verbose == True:
                                            print("skipping rift: cannot confidently locate both walls")
                                            # could default to initial estimate with low confidence here
                            
                                else: # points within reasonable distance meet wall threshold only on one side
                                    if verbose == True:
                                        print("skipping rift: cannot confidently locate both walls")
                                        # could default to initial estimate with low confidence here

                            else:
                                # Rift walls do not exceed the threshold for medium/high confidence detection
                                # then we default to the initial rift width from the are below
                                # 50% (or chosen threshold) of running mean
                                # ----------
                                # see Halloween Crack case study information above
                                # ----------
                                
                                confidence = "low"
                                    
                                # calculate the coordinates of the center of the rift (polar stereographic)
                                rift_centroid_x = (dist_start_x + dist_end_x) / 2
                                rift_centroid_y = (dist_start_y + dist_end_y) / 2

                                # calculate the coordinates of the center of the rift (lat/lon)
                                [rift_centroid_lat,rift_centroid_lon] = \
                                transformer.transform(rift_centroid_x,rift_centroid_y)

                                # calculate rift width
                                rift_width = dist_end - dist_start
                                    
                                seaward_d = np.nan
                                landward_d = np.nan
                                seaward_h = np.nan
                                landward_h = np.nan
                                sl_offset = np.nan
                                    
                                rift_azimuth = azimuth[rift_start:rift_end+1].mean()
                                rift_sigma = sigma[rift_start:rift_end+1].mean()
                                rift_h = np.nan
                                    
                                rift_obs["d-start"].append(dist_start)
                                rift_obs["d-end"].append(dist_end)
                                rift_obs["x-start"].append(dist_start_x)
                                rift_obs["y-start"].append(dist_start_y)
                                rift_obs["x-end"].append(dist_end_x)
                                rift_obs["y-end"].append(dist_end_y)
                                rift_obs["x-centroid"].append(rift_centroid_x)
                                rift_obs["y-centroid"].append(rift_centroid_y)
                                rift_obs["lat-centroid"].append(rift_centroid_lat)
                                rift_obs["lon-centroid"].append(rift_centroid_lon)
                                rift_obs["width"].append(rift_width)
                                rift_obs["d_seaward"].append(seaward_d)
                                rift_obs["d_landward"].append(landward_d)
                                rift_obs["h_seaward"].append(seaward_h)
                                rift_obs["h_landward"].append(landward_h)
                                rift_obs["sl_offset"].append(sl_offset)
                                rift_obs["time"].append(time)
                                rift_obs["rgt"].append(rgt)
                                rift_obs["azimuth"].append(rift_azimuth)
                                rift_obs["sigma"].append(rift_sigma)
                                rift_obs["h"].append(rift_h)
                                rift_obs["beam"].append(beam)
                                rift_obs["data_row"].append(i)
                                rift_obs["confidence"].append(confidence)    
                                
                                if verbose == True:
                                    print("cannot refine rift measurements")
                                    print("defaulting to initial estimate")
                                    print("confidence: low")
                                    

                        else: # all points of given rift are masked
                            if verbose == True:
                                print("skipping rift: rift detection in masked region")
                                print("or all points flagged as poor quality")
                
                    else: # rift end = end of trace
                        if verbose == True:
                            print("skipping rift: possible calving front")
                
                else: # rift start = 0
                    if verbose == True:
                        print("skipping rift: possible grounding line rift")
                        # we will come back to this. We don't want to skip rifts at the grounding line
                        # but for now the ATL06 is clipped to ice shelf shapefiles
                
        else: # rift detector didn't find any rifts 
            if verbose == True:
                print("rift detector did not find any rifts")
                       
        if verbose == True:
            print("------------------------------")
            print("rift detection and measurement complete for trace")
            print("rgt : "+str(rgt))
            print("beam: "+beam)
            print("date: "+str(time))
        
    rift_obs = pd.DataFrame(rift_obs)
    nr_rifts_final = len(rift_obs)
    
    #if verbose == True:
    print("------------------------------")
    print("")
    print("rift detector found and measured a total of "+str(nr_rifts_final)+" rifts")
    
    ttend = t.perf_counter()
    print("")
    print("time to detect rifts: "+str(round((ttend - ttstart)/60, 1))+" minutes")
    
    return(rift_obs)    








