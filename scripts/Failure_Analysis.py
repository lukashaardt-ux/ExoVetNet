import glob
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch

from ExoVetNet import Net, cnn_eval, plot_cm


def has_gap_artifact(local_view, min_run=5):

    is_zero = (local_view == 0.0)
    max_run = 0
    run = 0
    for v in is_zero:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run >= min_run

BATCH_SIZE = 64

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

cnn = Net().to(device)
cnn.load_state_dict(torch.load('model/ExoVetNet.pt'))
cnn.eval()

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


# Shuffling lists to avoid bias and increase generalization

indices = list(range(len(g_views)))
np.random.seed(42)
np.random.shuffle(indices)

g_views = [g_views[i] for i in indices]
l_views = [l_views[i] for i in indices]
labels = [labels[i] for i in indices]
names = [names[i] for i in indices]

# Loss Function
loss_function = torch.nn.BCEWithLogitsLoss()

# Training and Testing sets

X_g = torch.tensor(np.array(g_views),dtype=torch.float32).view(-1,1,2001)
X_l = torch.tensor(np.array(l_views),dtype=torch.float32).view(-1,1,201)
y = torch.tensor(np.array(labels),dtype=torch.float32)

HOLDOUT_PCT = 0.3
train_size = int(len(y)*HOLDOUT_PCT)
test_size = int(train_size/2)

test_X_g = X_g[-test_size:]
test_X_l = X_l[-test_size:]
test_y = y[-test_size:]

test_names = names[-test_size:]

test_y_true = test_y.cpu().numpy()

f1, predictions, probs = cnn_eval(BATCH_SIZE, cnn, device, test_X_g, test_X_l, test_y_true)

cm = sklearn.metrics.confusion_matrix(test_y_true, predictions)
plot_cm(cm)

missed_planets = [test_names[i] for i in range(len(test_y_true))
                if test_y_true[i]==1 and predictions[i]==0]
false_alarms  = [test_names[i] for i in range(len(test_y_true))
                if test_y_true[i]==0 and predictions[i]==1] 
caught_planets = [test_names[i] for i in range(len(test_y_true))
                  if test_y_true[i]==1 and predictions[i]==1]

feat = pd.read_csv("data/exoplanet_features_analysis.csv")
missed = feat[feat["name"].isin(missed_planets)]

print("Missed")
print(missed.describe())

print("All Planets:")
print(feat[feat["label"]==1].describe())

'''
for name in missed_planets[:10]:
    print(name)
    flux_file = f"data/fluxes_catalogv2/{name}.npz"
    with np.load(flux_file) as d:
        g_view = d["global_view"]
        l_view = d["local_view"]

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(g_view)
    plt.title(f"Global View: {name}")
    plt.subplot(1,2,2)
    plt.plot(l_view)
    plt.title(f"Local View: {name}")
    plt.show()
'''

caught_artifact = 0
for name in caught_planets:
    with np.load(f"data/fluxes_catalogv2/{name}.npz") as d:
        if has_gap_artifact(d["local_view"]):
            caught_artifact += 1
print(f"missed: {len(missed_planets)}/{len(caught_planets)+len(missed_planets)} = {len(missed_planets)/(len(caught_planets)+len(missed_planets)):.1%}")
print(f"caught: {caught_artifact}/{len(caught_planets)} = {caught_artifact/len(caught_planets):.1%}")