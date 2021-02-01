import rasterio
import sys
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import Window

#fp = r'/Users/bradlipovsky/Dropbox/MilneIceShelf/Planet_20200729_SR_reproj.tif'
fp = r'/Users/lipovsky/Downloads/riise_2020028_1615_modis_ch02.tif'
raster = rasterio.open(fp)

fig = plt.figure()
ax = fig.add_subplot(111)
show(raster,ax=ax,cmap='gray')
window_x = [-7.2e5,-6.4e5]
window_y = [1.40e6,1.46e6]
plt.ylim(window_y)
plt.xlim(window_x)
#plt.set_cmap('binary')


import os
import pickle
all_polygons = []
pth = '../BendShelf/'
for root, dirs, files in os.walk(pth):
	for f in files:
		if f.endswith(".pickle"):
			hf = pickle.load( open( pth+f, "rb" ) )
			x,y = hf.exterior.xy
			plt.plot(x,y,c='red')
	
coords = []
ind = 0
def onclick(event):
	global ind
	
	# Only add the coordinate after the first click 
	# (hack to allow the first click to operate the zoom button)
	if ind > 0:
		global ix, iy
		ix, iy = event.xdata, event.ydata
		print ('x = %d, y = %d'%(ix, iy))
		plt.scatter(ix,iy,c='r')
		plt.draw()

		global coords
		coords.append((ix, iy))

	else:
		ind = 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)


from shapely.geometry import Point, LineString
poly = LineString(coords)

# Save a dictionary into a pickle file.
import pickle
pickle.dump( poly, open( "halloween.pickle", "wb" ) )
