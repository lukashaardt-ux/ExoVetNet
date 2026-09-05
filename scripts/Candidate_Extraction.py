import os

import lightkurve as lk
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from tqdm import tqdm

USE_BLS = False # Turn on to extract with BLS periodogram rather than archive data

# Period, epoch, duration, kepid, name data

df = pd.read_csv("data/candidate_features.csv")

df_copy = df.copy()

# == Flux array extraction loop ==

# Flux array folder
if not USE_BLS:
    os.makedirs("data/candidate_fluxes_catalog",exist_ok=True)
else:
    os.makedirs("data/candidate_fluxes_bls", exist_ok=True)


for kepid in tqdm(df["kepid"].unique()):

    star_names = df_copy[df_copy["kepid"] == kepid]["name"]

    if len(star_names) > 0 and all(os.path.exists(f"data/candidate_fluxes_{ 'bls' if USE_BLS else 'catalog' }/{n}.npz") for n in star_names):
        continue

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

            out_dir = "data/candidate_fluxes_bls" if USE_BLS else "data/candidate_fluxes_catalog"

            if os.path.exists(f"{out_dir}/{name}.npz"):
                continue

            if not USE_BLS:

                period = df_copy[df_copy["kepid"] == kepid]["period"].iloc[0]
                epoch = df_copy[df_copy["kepid"] == kepid]["epoch"].iloc[0]
                duration = df_copy[df_copy["kepid"] == kepid]["duration"].iloc[0]

            else:

                cat_period = df_copy[df_copy["kepid"] == kepid]["period"].iloc[0]
                cat_duration = df_copy[df_copy["kepid"] == kepid]["duration"].iloc[0]


                pg = flat_lc.to_periodogram(method="bls",minimum_period = cat_period*0.9, maximum_period = cat_period*1.1, frequency_factor = 500, duration = np.arange(cat_duration*0.5, cat_duration*1.5, cat_duration*0.1))


                period = pg.period_at_max_power.value
                epoch = pg.transit_time_at_max_power.value
                duration = pg.duration_at_max_power.value

            folded_lc = flat_lc.fold(period, epoch_time=epoch)

            # Local (Around the transit)


            p_width = (3*duration)

            phase = folded_lc.time.value
            flux = folded_lc.flux.value

            mask = ((np.abs(phase)) < p_width/2)
            p_masked = phase[mask]
            f_masked = flux[mask]

            # = Flux array creation =

            # Global flux array
            
            flux_arr_g, _, _ = binned_statistic(phase, flux, statistic="median",
                                                bins = 2001, range = (phase.min(), phase.max()))

            flux_arr_g = flux_arr_g - np.nanmedian(flux_arr_g)

            flux_arr_g = flux_arr_g / np.abs(np.nanmin(flux_arr_g))

            flux_arr_g = np.asarray(flux_arr_g, dtype=float)

            flux_arr_g = np.nan_to_num(flux_arr_g, nan=0.0)

            # Local flux array
            
            flux_arr_l, _, _ = binned_statistic(p_masked, f_masked, statistic="median",
                                    bins=201, range=(-p_width/2, p_width/2))            


            flux_arr_l = flux_arr_l - np.nanmedian(flux_arr_l)

            flux_arr_l = flux_arr_l / np.abs(np.nanmin(flux_arr_l))

            flux_arr_l = np.asarray(flux_arr_l, dtype=float)

            flux_arr_l = np.nan_to_num(flux_arr_l, nan=0.0)

            # Uploading values


            if not USE_BLS:
                np.savez(f"data/candidate_fluxes_catalog/{name}.npz",
                        global_view = flux_arr_g,
                        local_view = flux_arr_l,
                )

            else:
                np.savez(f"data/candidate_fluxes_bls/{name}.npz",
                        global_view = flux_arr_g,
                        local_view = flux_arr_l,
                )

        except Exception as e:  # noqa: BLE001
            print(f"Error occurred while processing KIC {kepid}: {e}")

        finally:
            match = df_copy[df_copy["kepid"] == kepid].index
            if len(match) > 0:
                df_copy = df_copy.drop(match[0])
