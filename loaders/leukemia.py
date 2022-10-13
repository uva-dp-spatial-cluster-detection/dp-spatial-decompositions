from . import zcta

import pathlib
import pandas as pd

DEFAULT_DATA_PATH = pathlib.Path("data/zip_lvl_leukemia_population_2020.txt")

def load_data(csv_path: pathlib.Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    geo_df = zcta.load_data()
    df = pd.read_csv(csv_path, sep=" ", index_col="ZIP").rename(columns={"Population_with_leukemia_2020": "count"})
    df.index = df.index.rename("zcta")
    return pd.merge(geo_df, df, how="right", left_index=True, right_index=True)
