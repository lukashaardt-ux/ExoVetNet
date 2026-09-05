import glob
import os

import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from torch import nn, optim
from tqdm import tqdm

PLANET_VIEW = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Global input
        
        self.convg1 = nn.Conv1d(1,32,3)
        self.convg2 = nn.Conv1d(32,64,3)
        self.convg3 = nn.Conv1d(64,128,3)
        self.convg4 = nn.Conv1d(128,256,3)


        # Local input
        self.convl1 = nn.Conv1d(1,64,3)
        self.convl2 = nn.Conv1d(64,128,3)

        # Dropout (Convolutional Global)

        self.dropoutcon1g = nn.Dropout(0.3)
        self.dropoutcon2g = nn.Dropout(0.3)
        self.dropoutcon3g = nn.Dropout(0.3)

        # Dropout (Convolutional Local)
        
        self.dropoutcon1l = nn.Dropout(0.3)

        # Calculating FC layer input

        self.to_linearg = None
        self.to_linearl = None

        xg = torch.randn(1,1,2001)
        xl = torch.randn(1,1,201)

        self.convsg(xg)
        self.convsl(xl)

        # Fully connected layers
        self.fc1 = nn.Linear(self.to_linearg + self.to_linearl,512)
        self.fc2 = nn.Linear(512,1)

        # Dropout (FC Layer)

        self.dropoutfc = nn.Dropout(0.5)

    def convsg(self,xg):

        xg = F.max_pool1d(F.relu(self.convg1(xg)), 2)
        xg = self.dropoutcon1g(xg)
        xg = F.max_pool1d(F.relu(self.convg2(xg)), 2)
        xg = self.dropoutcon2g(xg)
        xg = F.max_pool1d(F.relu(self.convg3(xg)), 2)
        xg = self.dropoutcon3g(xg)
        xg = F.max_pool1d(F.relu(self.convg4(xg)), 2)

        if self.to_linearg == None:
            self.to_linearg = xg[0].shape[0]*xg[0].shape[1]

        return xg

    
    def convsl(self,xl):

        xl = F.max_pool1d(F.relu(self.convl1(xl)), 2)
        xl = self.dropoutcon1l(xl)
        xl = F.max_pool1d(F.relu(self.convl2(xl)), 2)

        if self.to_linearl == None:
            self.to_linearl = xl[0].shape[0]*xl[0].shape[1]

        return xl

    def forward(self,xg, xl):

        xg = self.convsg(xg)
        xl = self.convsl(xl)

        xg = xg.view(xg.size(0),-1)
        xl = xl.view(xl.size(0),-1)

        combined = torch.cat([xg,xl],dim=1)

        x = F.relu(self.fc1(combined))
        x = self.dropoutfc(x)
        x = self.fc2(x)

        return x 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

cnn = Net().to(device)
cnn.load_state_dict(torch.load("model/ExoVetNet.pt"))
cnn.eval()

flux_files = sorted(glob.glob("data/fluxes_catalogv2/*.npz"))

labels = []
g_views = [] # global views
l_views = [] # local views
names = [os.path.splitext(os.path.basename(f))[0] for f in flux_files]

for f in flux_files:
    #path = f"data/fluxes_catalog/{name}.npz"
    #if os.path.exists(path):
    with np.load(f) as d:
        labels.append(d["label"])
        g_views.append(d["global_view"])
        l_views.append(d["local_view"])


# Shuffling lists to avoid bias and increase generalization
np.random.seed(42)
indices = list(range(len(g_views)))
np.random.shuffle(indices)

g_views = [g_views[i] for i in indices]
l_views = [l_views[i] for i in indices]
labels = [labels[i] for i in indices]
names = [names[i] for i in indices]

# Optimizer

optimizer = optim.Adam(cnn.parameters(), lr = 0.001, weight_decay = 1e-4)

# Loss Function
loss_function = nn.BCEWithLogitsLoss()

# Training and Testing sets



X_g = torch.tensor(np.array(g_views),dtype=torch.float32).view(-1,1,2001)
X_l = torch.tensor(np.array(l_views),dtype=torch.float32).view(-1,1,201)
y = torch.tensor(np.array(labels),dtype=torch.float32)

HOLDOUT_PCT = 0.3
train_size = int(len(y)*HOLDOUT_PCT)
test_size = int(train_size/2)


X_g = X_g[-test_size:]
X_l = X_l[-test_size:]
y = y[-test_size:]

g_mean = X_g.mean(dim=0)
l_mean = X_l.mean(dim=0)

X_g_1 =  X_g[:1]
X_l_1 =  X_l[:1]
y1 = y[:1]

with torch.no_grad():
    outputs = cnn(X_g.to(device), X_l.to(device))

    probs = torch.sigmoid(outputs) 

    predicted_classes = (probs > 0.5).int().cpu().numpy().flatten()

print(predicted_classes)

#with open("model/ExoVetNet_provenance.txt", "w") as f:
#    f.write(f"folder=data/fluxes_catalogv2  n_files={len(flux_files)}  saved={time.time()}\n")


cap_inputs = (
    X_g_1.view(1, 1, 2001).to(device),
    X_l_1.view(1, 1, 201).to(device)
)

g_base = g_mean.view(1, 1, 2001).to(device)
l_base = l_mean.view(1, 1, 201).to(device)

baseline = (
    g_base,
    l_base
)

ig = captum.attr.IntegratedGradients(cnn)

attributions_ig = ig.attribute(cap_inputs,
                               baselines=baseline
                               )

global_attr, local_attr = attributions_ig

print("Global attribution shape:", global_attr.shape)
print("Local attribution shape:", local_attr.shape)

print("Global attribution range:",
      global_attr.min().item(),
      global_attr.max().item())

print("Local attribution range:",
      local_attr.min().item(),
      local_attr.max().item())



global_attr_np = global_attr.detach().cpu().numpy()[0, 0]
global_flux_np = X_g_1.detach().cpu().numpy()[0, 0]

local_attr_np = local_attr.detach().cpu().numpy()[0, 0]
local_flux_np = X_l_1.detach().cpu().numpy()[0, 0]

local_attrs_caught = []
local_attrs_missed = []


if PLANET_VIEW:

    missed_p_X_g = torch.Tensor([])
    missed_p_X_l = torch.Tensor([])


    for i, (g_view, l_view, y_label) in enumerate(zip(X_g, X_l, y)):
        if y_label == 1 and predicted_classes[i] == 0: # Confirmed Planet

            missed_p_X_g = torch.cat([missed_p_X_g, g_view.view(1, 1, 2001)], dim=0)
            missed_p_X_l = torch.cat([missed_p_X_l, l_view.view(1, 1, 201)], dim=0)


    for i in tqdm(range(len(missed_p_X_g))):

        xg = missed_p_X_g[i:i+1].to(device)
        xl = missed_p_X_l[i:i+1].to(device)

        attr = ig.attribute((xg, xl), baselines=(g_base, l_base))
        local_attrs_missed.append(attr[1].detach().cpu().numpy()[0,0]) 

    caught_p_X_g = torch.Tensor([])
    caught_p_X_l = torch.Tensor([])


    for i, (g_view, l_view, y_label) in enumerate(zip(X_g, X_l, y)):
        if y_label == 1 and predicted_classes[i] == 1:

            caught_p_X_g = torch.cat([caught_p_X_g, g_view.view(1, 1, 2001)], dim=0)
            caught_p_X_l = torch.cat([caught_p_X_l, l_view.view(1, 1, 201)], dim=0)


    for i in tqdm(range(len(caught_p_X_g))):

        xg = caught_p_X_g[i:i+1].to(device)
        xl = caught_p_X_l[i:i+1].to(device)

        attr = ig.attribute((xg, xl), baselines=(g_base, l_base))
        local_attrs_caught.append(attr[1].detach().cpu().numpy()[0,0]) 

local_attrs_caught = np.array(local_attrs_caught)
local_attrs_missed = np.array(local_attrs_missed)       

mean_local_attr_caught = local_attrs_caught.mean(axis=0)  
mean_local_attr_missed = local_attrs_missed.mean(axis=0)  

'''
plt.figure(figsize=(10, 5))
plt.plot(mean_local_attr, c="red", alpha=0.7)

if PLANET_VIEW:
    plt.title("Mean Local Attribution for Planetary Candidates")
    plt.savefig(f"figures/mean_local_attr_confirmed_planets{time.time()}.png", dpi=300)
if not PLANET_VIEW:
    plt.title("Mean Local Attribution for False Positives")
    plt.savefig(f"figures/mean_local_attr_false_positives{time.time()}.png", dpi=300)
'''

fig, axs = plt.subplots(2, figsize=(14,8), sharey=True)

axs[0].plot(mean_local_attr_caught, c="navy")
axs[0].set_ylabel("Integrated Gradients Attribution")
axs[0].set_title("Caught Exoplanets")
axs[0].grid(True)

axs[0].axvspan(70, 130, color='red', alpha=0.07)

axs[1].plot(mean_local_attr_missed, c="navy")
axs[1].set_xlabel("Local view bin (transit centered at ~100)")
axs[1].set_ylabel("Integrated Gradients Attribution")
axs[1].set_title("Missed Exoplanets")
axs[1].grid(True)

axs[1].axvspan(70, 130, color='red', alpha=0.07)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

fig.suptitle("ExoVetNet's attention on planets it catches is 2.7x more consistent than on planets it misses", fontsize=14, y=0.98)

plt.savefig("figures/missedvcaught.png")

plt.show()

window = slice(70, 131)
caught_focus = np.abs(local_attrs_caught[:, window]).mean(axis=1)
missed_focus = np.abs(local_attrs_missed[:, window]).mean(axis=1)

u, p = mannwhitneyu(caught_focus, missed_focus, alternative="greater")

print(f"caught: n={len(caught_focus)}  median={np.median(caught_focus):.4f}")
print(f"missed: n={len(missed_focus)}  median={np.median(missed_focus):.4f}")
print(f"ratio of medians: {np.median(caught_focus)/np.median(missed_focus):.2f}x")
print(f"Mann-Whitney p={p:.3e}   P(random caught > random missed)={u/(len(caught_focus)*len(missed_focus)):.3f}")
c_caught = np.abs(local_attrs_caught[:, window].mean(axis=0)).mean() / np.abs(local_attrs_caught[:, window]).mean()
c_missed = np.abs(local_attrs_missed[:, window].mean(axis=0)).mean() / np.abs(local_attrs_missed[:, window]).mean()
print(f"coherence — caught: {c_caught:.3f}  missed: {c_missed:.3f}  ratio: {c_caught/c_missed:.2f}x")
'''
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1, ax2, ax3, ax4 = axes.flatten()

ax1.plot(global_flux_np, c="navy",alpha=0.7)

ax1.set_xlabel("Time index")
ax1.set_ylabel("Normalized flux")
ax1.set_title("ExoVetNet Global Light Curve")

ax2.plot(global_attr_np, c="red",alpha=0.7)

ax2.axhline(0, linestyle="--", c="crimson", alpha=0.5)

ax2.set_xlabel("Time index")
ax2.set_ylabel("Integrated Gradients Attribution")
ax2.set_title("ExoVetNet Global Attribution")

ax3.plot(local_flux_np, c="navy",alpha=0.7)

ax3.set_xlabel("Time index")
ax3.set_ylabel("Normalized flux")
ax3.set_title("ExoVetNet Local Light Curve")

ax4.plot(local_attr_np, c="red",alpha=0.7)

ax4.set_xlabel("Time index")
ax4.set_ylabel("Integrated Gradients Attribution")
ax4.set_title("ExoVetNet Local Attribution")

plt.show()
'''