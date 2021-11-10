import icepyx as ipx
from time import perf_counter
import json
from arc import get_coords
import os

def main():
    datapath = '/data/fast1/arc/'

    t_start = perf_counter()
    shelf_names = ('brunt', 'fimbul', 'amery', 
                    'ap', 'ross', 'ronne', 'amundsen', 'east')
    date_range = ['2018-10-1','2021-12-31']
    short_name = 'ATL06'

    for shelf_name in shelf_names:
        s = get_coords(shelf_name)
        region_a = ipx.Query(short_name, s, date_range)
        granules=region_a.avail_granules(ids=True)

        data = granules[0]
        file = os.path.join(datapath,'filelists/', shelf_name + '-list.json')
        with open(file, 'w') as outfile:
            json.dump(data, outfile)
        print('Wrote file %s'%file)
        print(perf_counter()-t_start)

if __name__ == "__main__":
    main()