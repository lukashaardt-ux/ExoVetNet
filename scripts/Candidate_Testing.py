import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import style
from torch import nn


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

# main
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

cnn = Net().to(device)
cnn.load_state_dict(torch.load("model/ExoVetNet.pt"))
cnn.eval()

flux_files = glob.glob("data/candidate_fluxes_catalog/*.npz")

g_views = []
l_views = []
names = [os.path.splitext(os.path.basename(f))[0] for f in flux_files]

for f in flux_files:

    with np.load(f) as data:
        g_view = data['global_view']
        l_view = data['local_view']

        g_views.append(g_view)
        l_views.append(l_view)

indices = list(range(len(g_views)))
np.random.shuffle(indices)

g_views = [g_views[i] for i in indices]
l_views = [l_views[i] for i in indices]
names = [names[i] for i in indices]

X_g = torch.tensor(np.array(g_views),dtype=torch.float32).view(-1,1,2001).to(device)
X_l = torch.tensor(np.array(l_views),dtype=torch.float32).view(-1,1,201).to(device)

predictions = []
probs = []

with torch.no_grad():
    for i in range(len(X_g)):
    
        g = X_g[i].view(1,1,2001)
        l = X_l[i].view(1,1,201)

        output = cnn(g,l)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0

        predictions.append(pred)
        probs.append(prob)

style.use('ggplot')

# Histogram of probabilities

sns.histplot(probs, bins=30, kde=True, alpha=0.7, color='purple', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities for Exoplanet Candidates')
plt.savefig('figures/candidate_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Barplot of predictions

sns.countplot(x=predictions, alpha=0.7, color='purple', edgecolor='black')
plt.xticks([0, 1], ['Non-Exoplanet', 'Exoplanet'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions for Exoplanet Candidates')
plt.savefig('figures/candidate_predictions.png', dpi=300, bbox_inches='tight')
plt.show()