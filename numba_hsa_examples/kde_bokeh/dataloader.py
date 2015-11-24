import h5py
import pandas as pd
from functools import reduce
import numpy as np
import os
import numba.hsa

from numba_hsa_examples.pandas_eval import eval_engine

eval_engine.register()

basedir = os.path.dirname(__file__)

MIN_RAD = 30000

QUERY_ENGINE = "numba.hsa" if numba.hsa.is_available() else "numba.cpu"


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
    """
    Filter a dataframe to only include data points within the given criteria
    """
    lat_in_range = "(lat >= @lat_min and lat <= @lat_max)"
    lon_in_range = "(lon >= @lon_min and lon <= @lon_max)"
    rad_in_range = "(rad >= @rad_min)"
    conditions = lat_in_range, lon_in_range, rad_in_range
    query = ' and '.join(conditions)
    out = df.query(query, engine=QUERY_ENGINE)
    print("reduce data elements from {0} to {1}".format(df.size, out.size))
    return out


def load_all_data(lon_min, lon_max, lat_min, lat_max):
    """
    Returns all the data points in a DataFrame
    """
    filenames = ['data/events.hdf5']
    files = [os.path.join(basedir, f) for f in filenames]

    dfs = (load_file_as_dataframe(f) for f in files)
    criteria = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
                    lon_max=lon_max)
    filtered = (filter_dataframes(df, **criteria) for df in dfs)
    return reduce(lambda a, b: a.append(b), filtered)


def downsample_dataset():
    """
    Used to create the "data/events.hdf5" dataset from the original full
    dataset.
    """
    df = load_all_data(-125, -65, 25, 50)
    rad = df.rad.values
    print(rad.size)
    print('mean', np.mean(rad), '; std', np.std(rad))

    f = h5py.File('myfile.hdf5', 'w')
    g = f.create_group('events')
    attrs = dict(compression="gzip", compression_opts=9, shuffle=True)
    g.create_dataset("latitude", data=df.lat.values, **attrs)
    g.create_dataset("longitude", data=df.lon.values, **attrs)
    g.create_dataset("radiance", data=df.rad.values, **attrs)


if __name__ == '__main__':
    downsample_dataset()
