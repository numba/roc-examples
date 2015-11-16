import h5py
import pandas as pd
from functools import reduce
import numpy as np
import os

basedir = os.path.dirname(__file__)

# MIN_RAD = 14711.8
# MIN_RAD = 37324.5
MIN_RAD = 30000


def load_file_as_dataframe(path):
    print("load", path)
    datfile = h5py.File(path)
    evts = datfile['events']
    df = pd.DataFrame({'lat': evts['latitude'],
                       'lon': evts['longitude'],
                       'rad': evts['radiance']})
    return df


def filter_dataframes(df, lat_min, lat_max, lon_min, lon_max,
                      rad_min=MIN_RAD):
    print("filter")
    lat_in_range = (df.lat >= lat_min) & (df.lat <= lat_max)
    lon_in_range = (df.lon >= lon_min) & (df.lon <= lon_max)
    rad_in_range = df.rad > rad_min
    out = df[lat_in_range & lon_in_range & rad_in_range]
    print("reduce from {0} to {1}".format(df.size, out.size))
    return out


def load_all_data(lon_min, lon_max, lat_min, lat_max):
    # files = ["data/event_00.hdf5",
    #          "data/event_15.hdf5"]
    filenames = ["data/event_{0:02d}.hdf5".format(i) for i in range(5)]

    files = [os.path.join(basedir, f) for f in filenames]

    dfs = (load_file_as_dataframe(f) for f in files)
    criteria = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
                    lon_max=lon_max)
    filtered = (filter_dataframes(df, **criteria) for df in dfs)
    return reduce(lambda a, b: a.append(b), filtered)


if __name__ == '__main__':
    df = load_all_data(-125, -65, 25, 50)
    rad = df.rad.values
    print(rad.size)
    print(np.mean(rad), np.std(rad))
