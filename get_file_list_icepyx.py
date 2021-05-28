import icepyx as ipx
from time import perf_counter
import json
from arc import get_coords

t_start = perf_counter()
region_names = ('brunt', 'fimbul', 'amery', 
                'ap', 'ross', 'ronne', 'amundsen', 'east')
date_range = ['2018-10-1','2021-12-31']
short_name = 'ATL06'

for n in region_names:
    s = get_coords(n)
    region_a = ipx.Query(short_name, s, date_range)
    granules=region_a.avail_granules(ids=True)

    data = granules[0]
    file = './filelists/%s-list.json'%n
    with open(file, 'w') as outfile:
        json.dump(data, outfile)
    print('Wrote file %s'%file)
    print(perf_counter()-t_start)
