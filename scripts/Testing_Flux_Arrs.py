import glob

import numpy as np

#d = np.load("data/fluxes/K00752.01.npz")
#print(d["global_view"].shape, d["local_view"].shape, d["label"])
#print(np.isnan(d["global_view"]).any(), np.isnan(d["local_view"]).any())
#plt.plot(d["global_view"]); plt.show()
#plt.plot(d["local_view"]); plt.show()

#a = np.load("data/fluxes/K00756.01.npz")["local_view"]
#b = np.load("data/fluxes/K00756.02.npz")["local_view"]
#c = np.load("data/fluxes/K00756.03.npz")["local_view"]
#print(np.allclose(a, b), np.allclose(a, c))   # both should be False
flux_files = glob.glob("data/fluxes_catalog/*.npz")

labels=[]

for f in flux_files[:3622]:
    with np.load(f) as d:
        labels.append(str(d["label"]))

# Count frequencies
label_dict = {}
for label in labels:
    if label in label_dict:
        label_dict[label] += 1
    else:
        label_dict[label] = 1

# Print percentages accurately
for label, count in label_dict.items():
    percentage = (count / len(labels)) * 100
    print(f"{label}: {round(percentage, 2)}%")