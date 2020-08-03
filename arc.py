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

    atl06_data = {"lat":list(),"lon":list(),"h":list(),"azimuth":list(),
                  "h_sig":list(),"rgt":list(),"acquisition_number":list(),"time":list(),
                  "x":list(), "y":list() }

    acq = 0
    file_index = 1

    nf = len(files)
    for f in files:
        FILE_NAME = os.path.join(data_directory,f)
        fid = h5py.File(FILE_NAME, mode='r')

        for lr in ("l","r"):
            for i in range(1,4):
                try:
                    h_li = fid['gt%i%s/land_ice_segments/h_li'%(i,lr)][:]
                    h_lat = fid['gt%i%s/land_ice_segments/latitude'%(i,lr)][:]
                    h_lon = fid['gt%i%s/land_ice_segments/longitude'%(i,lr)][:]
                    h_li_sigma = fid['gt%i%s/land_ice_segments/h_li_sigma'%(i,lr)][:]
                    seg_az = fid['gt%i%s/land_ice_segments/ground_track/seg_azimuth'%(i,lr)][:]
                    rgt = fid['/orbit_info/rgt'][0]
                    time = dparser.parse( fid['/ancillary_data/data_start_utc'][0] ,fuzzy=True )
                    [h_x,h_y] = transformer.transform( h_lat , h_lon )

                except KeyError:
        #                 print("wtf key error")
                    continue

#             This is just used for Brunt:
                if any(h_lon>0):
#                         Not clear why some of the data is out of the region of interest
                    continue
                else:

                    atl06_data["lat"].append( h_lat )
                    atl06_data["lon"].append( h_lon )
                    atl06_data["x"].append( h_x )
                    atl06_data["y"].append( h_y )
                    atl06_data["h_sig"].append( h_li_sigma )
                    atl06_data["h"].append( h_li)
                    atl06_data["azimuth"].append( seg_az )
                    atl06_data["rgt"].append( [rgt]*len(h_li) )
                    atl06_data["time"].append( [time]*len(h_li) )
                    atl06_data["acquisition_number"].append( [acq]*len(h_li) )
                    acq = acq + 1
                
#         print('Loaded file #%i of %i'%(file_index,nf))
        file_index = file_index+1
        fid.close()

    ttend = datetime.now()
    print('Runtime was: ', ttend - ttstart)
    
    #
    # Now convert to polar stereo coordinates
    #
    
    


    # Store data (serialize)
    with open(output_file_name, 'wb') as handle:
        pickle.dump(atl06_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
        
        
def find_the_rifts(trace):
    
    thr = 4
    
    # Melange is visible as near zero m above sea level
    melange = np.where(abs(trace) < thr) #absolute value because some erroneous points in ATL06 are large and negative
    melange = melange[0]
    
    if len(melange) < 2:
        return []
    
    start = np.nan
    stop = np.nan
    
    segments=[]
    mx=np.max(melange)

    # now create a list with the format ((seg1start,seg1stop),(seg2start,seg2stop),...)
    for i in range(len(melange)):
        if i == 0:
            start = melange[i]
            
        if i == len(melange)-1:
            stop = melange[i]
            segments.append((start,stop))
            return segments
        
        if melange[i+1] > melange[i]+1:
            stop = melange[i]
            segments.append((start,stop))
#             print("Added {strt} , {stp}".format(strt=start,stp=stop))
            start = melange[i+1]
            stop = np.nan
            
    return segments
        
    
    
    
    
    
def convert_to_centroid(rift_list,x,y):
    centroid_x = list()
    centroid_y = list()
    width = list()
    
    for r in rift_list:
        centroid_x.append( (x[r[0]] + x[r[1]])/2 )
        centroid_y.append( (y[r[0]] + y[r[1]])/2 )
        width.append( np.sqrt((x[r[0]] - x[r[1]])**2 + (y[r[0]] - y[r[1]])**2) )
        
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
        "acq":[],
        "h":[]
    }


    ttstart = datetime.now()

    for i, row in atl06_dataframe.iterrows():
        
        rift_list = find_the_rifts( row['h'] )

        if len(rift_list) > 0:
            
            rift_azi = []
            rift_sig = []
            rift_h = []
            for rift_coords in rift_list:
                rift_azi.append ( row['azimuth'][rift_coords[0]:rift_coords[1]].mean()   )
                rift_sig.append ( row['h_sig'][rift_coords[0]:rift_coords[1]].mean()  )
                rift_h.append   ( row['h'][rift_coords[0]:rift_coords[1]].mean()  )
            
            output = convert_to_centroid(rift_list,row['x'],row['y'])
            
            rift_obs['x-centroid'].extend( output['x-centroid'] )
            rift_obs['y-centroid'].extend( output['y-centroid'] )
            rift_obs['width'].extend( output['width'] )
            
            rift_obs['time'].extend( [ row['time'][0] ] * len(rift_list) )
            rift_obs['rgt'].extend( [ row['rgt'][0] ] * len(rift_list) )
            rift_obs['acq'].extend( [i] * len(rift_list) )
            
            rift_obs['azimuth'].extend ( rift_azi )
            rift_obs['sigma'].extend( rift_sig )
            rift_obs['h'].extend( rift_h )
            
    #     print('Completed %i of %i at %s'%(i,na,datetime.now()))

    # Save centroid locations in lat-lon
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
    [lon,lat] = transformer.transform(rift_obs['x-centroid'] , rift_obs['y-centroid'] )

    rift_obs['lat'] = lat
    rift_obs['lon'] = lon

    ttend = datetime.now()
    print('Run time:', ttend - ttstart)
    
    return rift_obs

def find_rifts_in_atl03(rift_obs):
    # To be written





