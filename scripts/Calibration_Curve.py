import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

BINS = 6
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

with torch.no_grad():
    outputs = cnn(X_g.to(device),X_l.to(device))

    probs = torch.sigmoid(outputs)

bin_bounds = np.linspace(0,1,BINS+1)

bin_indices = np.digitize(probs.cpu().numpy(),bin_bounds) - 1

mean_probs = []
mean_fracs = []

for i in tqdm(range(BINS)):

    bin_mask = bin_indices == i

    if bin_mask.sum() == 0:
        continue

    print(f"Bin {i}: {bin_mask.sum()} observations")

    bin_X = probs[bin_mask]
    bin_y = y[bin_mask.flatten()].cpu().numpy()

    frac_pos = bin_y.mean()
    mean_prob = bin_X.mean().cpu().numpy().astype(np.float32)

    print(type(mean_prob))
    print(type(frac_pos))

    mean_probs.append(mean_prob)
    mean_fracs.append(frac_pos)

print(type(mean_probs))
print(type(mean_fracs))

plt.plot(mean_probs, mean_fracs, c="#d62728", marker="o", label = "ExoVetNet")
plt.plot([0, 1], [0, 1], linestyle="--", label = "Reference Line")
plt.legend()
plt.grid(True)

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("ExoVetNet Calibration Curve")

plt.savefig("figures/calibration_curve.png")

plt.show()

brier = torch.mean((probs.cpu().flatten() - y) ** 2).item()

print(f"Brier Score: {brier:.4f}")