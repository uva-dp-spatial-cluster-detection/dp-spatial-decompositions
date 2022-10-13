"""Loads US ZIP Code Tabluation Areas (ZCTAs), such as those available at
https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
"""

import pathlib

import geopandas
from shapely import geometry

DEFAULT_DATA_PATH = pathlib.Path("data/tl_2022_us_zcta520.shp")

def load_data(shapefile_path: pathlib.Path = DEFAULT_DATA_PATH) -> geopandas.GeoDataFrame:
    """Loads .SHP/.SHX/.DBF files, such as those available at
    https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
    """
    # load files
    df_zcta = geopandas.read_file(shapefile_path)
    # rename columns and drop unneeded cols
    df_zcta = df_zcta.rename(columns={"ZCTA5CE20": "zcta"})[["zcta", "geometry"]]
    # convert each zcta from str to int
    df_zcta.zcta = df_zcta.zcta.astype(int)
    # use zctas as index
    df_zcta = df_zcta.set_index("zcta")
    return df_zcta

def to_shapely(df: geopandas.GeoDataFrame, zcta: int):
    shape = df.loc[zcta].geometry
    return geometry.shape(shape)
