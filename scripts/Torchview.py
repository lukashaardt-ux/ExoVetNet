from torchview import draw_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Requires Graphviz to be installed and its `bin` directory on PATH.
# If it isn't picked up automatically, set GRAPHVIZ_BIN to that directory.
graphviz_bin = os.environ.get("GRAPHVIZ_BIN")
if graphviz_bin:
    os.environ["PATH"] += os.pathsep + graphviz_bin


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

graph = draw_graph(
    cnn,
    input_size=[(1, 1, 2001), (1, 1, 201)],   
    device=device,
    expand_nested=True,
)

graph.visual_graph.render("figures/ExoVetNet_architecture", format="png")