import os

import lightkurve as lk
import numpy as np
import pandas as pd
from tqdm import tqdm

# Period, epoch, duration, kepid, name data

df = pd.read_csv("data/exoplanet_features.csv")

df_copy = df.copy()

# == BLS extraction loop ==


for kepid in tqdm(df["kepid"].unique()):

    star_names = df_copy[df_copy["kepid"] == kepid]["name"]

    try:
        search_result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
    except Exception as e:  # noqa: BLE001
        print(f"Error occurred while searching for light curve for KIC {kepid}: {e}")
        continue

    try:
        lc_collection = search_result.download_all()
    except Exception as e:  # noqa: BLE001
        print(f"Error occurred while downloading light curves for KIC {kepid}: {e}")
        continue


    try:
        flat = [lc.remove_nans().flatten().remove_outliers() for lc in lc_collection]

        flat_lc = lk.LightCurveCollection(flat).stitch()
    except Exception as e:  # noqa: BLE001
        print(f"Error occurred while flattening light curves for KIC {kepid}: {e}")
        continue
    

    while kepid in df_copy["kepid"].values:
        try:

            name = df_copy[df_copy["kepid"] == kepid]["name"].iloc[0]



            cat_period = df_copy[df_copy["kepid"] == kepid]["period"].iloc[0]
            cat_duration = df_copy[df_copy["kepid"] == kepid]["duration"].iloc[0]
            cat_epoch = df_copy[df_copy["kepid"] == kepid]["epoch"].iloc[0]


            pg = flat_lc.to_periodogram(method="bls",minimum_period = cat_period*0.9, maximum_period = cat_period*1.1, frequency_factor = 500, duration = np.arange(cat_duration*0.5, cat_duration*1.5, cat_duration*0.1))


            period = pg.period_at_max_power.value
            epoch = pg.transit_time_at_max_power.value
            duration = pg.duration_at_max_power.value


            label = df_copy[df_copy["kepid"] == kepid]["label"].iloc[0]

            row = pd.DataFrame([{
                "name" : name,
                "period" : period, "cat_period" : cat_period, 
                "epoch" : epoch, "cat_epoch" : cat_epoch, 
                "duration" : duration, "cat_duration" : cat_duration,
                "label" : label
            }])

            row.to_csv("data/BLSvCatalog_data.csv", mode="a",
                    header=not os.path.exists("data/BLSvCatalog_data.csv"),
                    index=False)            
        except Exception as e:  # noqa: BLE001
            print(f"{kepid} {e}")

        finally:
            match = df_copy[df_copy["kepid"] == kepid].index
            if len(match) > 0:
                df_copy = df_copy.drop(match[0])
