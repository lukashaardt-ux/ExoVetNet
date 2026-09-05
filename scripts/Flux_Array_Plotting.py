import glob
import os

import matplotlib.pyplot as plt
import numpy as np

flux_files = sorted(glob.glob("data/fluxes_catalogv2/*.npz"))

labels = []
g_views = [] # global views
l_views = [] # local views
names = [os.path.splitext(os.path.basename(f))[0] for f in flux_files]

print(names)

for f in flux_files:
    with np.load(f) as d:
        labels.append(d["label"])
        g_views.append(d["global_view"])
        l_views.append(d["local_view"])


for name in names:
    with np.load(f"data/fluxes_catalogv2/{name}.npz") as d:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(d["global_view"])
        plt.title(f"{name} - Global View")
        plt.subplot(1,2,2)
        plt.plot(d["local_view"])
        plt.title(f"{name} - Local View")
        plt.show()