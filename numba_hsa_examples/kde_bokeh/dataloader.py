import h5py
import pandas as pd
from functools import reduce
import os

basedir = os.path.dirname(__file__)


def load_file_as_dataframe(path):
    datfile = h5py.File(path)
    evts = datfile['events']
    df = pd.DataFrame({'lat': evts['latitude'],
                       'lon': evts['longitude'],
                       'rad': evts['radiance']})
    return df


def filter_dataframes(df, lat_min, lat_max, lon_min, lon_max):
    lat_in_range = (df.lat >= lat_min) & (df.lat <= lat_max)
    lon_in_range = (df.lon >= lon_min) & (df.lon <= lon_max)
    return df[lat_in_range & lon_in_range]


def load_all_data(lon_min, lon_max, lat_min, lat_max):
    # files = ["data/event_00.hdf5",
    #          "data/event_15.hdf5"]

    files = [os.path.join(basedir, "data/event_00.hdf5")]

    dfs = (load_file_as_dataframe(f) for f in files)
    criteria = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
                    lon_max=lon_max)
    filtered = (filter_dataframes(df, **criteria) for df in dfs)
    return reduce(lambda a, b: a.append(b), filtered)


if __name__ == '__main__':
    df = load_all_data()
    print(df)
