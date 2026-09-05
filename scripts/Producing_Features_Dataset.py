import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from tqdm import tqdm

known_data = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    select="kepid, kepoi_name, koi_time0bk, koi_period, koi_duration, koi_depth, koi_prad, koi_disposition"
)

names = []
labels = []
periods = []
epochs = []
durs = []
kepids = []
depths = []
p_rads = []

can_names = []
can_periods = []
can_epochs = []
can_durs = []
can_kepids = []

for i in tqdm(range(len(known_data))):

    label = known_data["koi_disposition"][i]


    if label == "CANDIDATE":
            can_kepids.append(known_data["kepid"][i])        
            can_names.append(known_data["kepoi_name"][i])
            can_periods.append(known_data["koi_period"][i].value)
            can_epochs.append(known_data["koi_time0bk"][i])
            can_durs.append(known_data["koi_duration"][i].to("day").value)
            continue
    else:
        label = 0 if label == "FALSE POSITIVE" else 1

    kepids.append(known_data["kepid"][i])        
    names.append(known_data["kepoi_name"][i])
    labels.append(label)
    periods.append(known_data["koi_period"][i].value)
    epochs.append(known_data["koi_time0bk"][i])
    durs.append(known_data["koi_duration"][i].to("day").value)
    depths.append(known_data["koi_depth"][i].value)
    p_rads.append(known_data["koi_prad"][i].value)

#df = pd.DataFrame({
#    "kepid": kepids, "name": names, "label": labels,
#    "period": periods, "epoch": epochs, "duration": durs
#})

#df.to_csv("exoplanet_features.csv", index=False)

can_df = pd.DataFrame({
    "kepid": can_kepids, "name": can_names,
    "period": can_periods, "epoch": can_epochs, "duration": can_durs
})

can_df.to_csv("data/candidate_features.csv", index=False)

print(len(can_df))
print(can_df.isna().sum())

df = pd.DataFrame({
    "kepid": kepids, "name": names, "label": labels,
    "period": periods, "epoch": epochs, "duration": durs,
    "depth": depths, "p_rad": p_rads
})

df.to_csv("data/exoplanet_features_analysis.csv", index=False)

print(len(df))
print(df["label"].value_counts())
print(df.isna().sum())
