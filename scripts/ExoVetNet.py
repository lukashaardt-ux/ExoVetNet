# Imports

import copy
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import style
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch import nn, optim
from tqdm import tqdm

# Constants

MODEL_NAME = f"ExoVetNet-{time.time()}"

EPOCHS = 20
BATCH_SIZE = 64

THRESHOLD = 0.50
NOISE_STD = 0.0



# Classes

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

# Functions

def fwd_pass(X1, X2, y, optimizer_pass, cnn_pass, train=False):

    if train:
        cnn_pass.zero_grad()

    outputs = cnn_pass(X1, X2)

    # Accuracy calculation

    correct = 0
    total = 0

    for i in range(len(X1)):

        prob  = torch.sigmoid(outputs[i])
        predicted_class = (prob > THRESHOLD).int()

        real_class = y[i]

        if predicted_class == real_class:
            correct +=1
        total +=1

    acc = correct/total

    # Loss calculation

    targets = y.float().unsqueeze(1)

    loss = loss_function(outputs, targets)

    if train:
        loss.backward()
        optimizer_pass.step()

    return acc, loss


def train(epochs, b_size):

    with open("data/ExoVetNet_log.csv", "a+") as f:

        #if len(f.read()) == 0:
        #    f.write("model_name,timestamp,epoch,step,accuracy,loss,validation_accuracy,validation_loss\n")
        
        step = 0
        for epoch in range(epochs):
            for i in tqdm(range(0, len(train_X_g), b_size)):

                cnn.train()

                batch_X_g = train_X_g[i:i+b_size].to(device)
                batch_X_l = train_X_l[i:i+b_size].to(device)

                batch_X_g = batch_X_g + torch.randn_like(batch_X_g) * NOISE_STD
                batch_X_l = batch_X_l + torch.randn_like(batch_X_l) * NOISE_STD


                batch_y = train_y[i:i+b_size].to(device)

                acc, loss = fwd_pass(batch_X_g, batch_X_l, batch_y, train=True)

                if i % 10 == 0:

                    step +=1
                    
                    val_acc, val_loss = validate(val_size_test=100)

                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{epoch},{step},{round(acc,2)},{round(float(loss),4)},{round(val_acc,2)},{round(float(val_loss),4)}\n")


def validate(val_size_test=32):

    randstart = np.random.randint(len(val_X_g) - val_size_test)

    X1, X2, y = val_X_g[randstart:randstart+val_size_test], val_X_l[randstart:randstart+val_size_test], val_y[randstart:randstart+val_size_test]

    cnn.eval()

    with torch.no_grad():

        val_acc, val_loss = fwd_pass(X1.to(device), X2.to(device), y.to(device))

    cnn.train()
    
    return val_acc, val_loss


def test(cnn_test, b_size):

    cnn_test.eval()

    test_predictions = []
    test_outputs = []

    with torch.no_grad():
        for i in range(0, len(test_X_g), b_size):

            xg = test_X_g[i:i+b_size].to(device)
            xl = test_X_l[i:i+b_size].to(device)

            outputs = cnn_test(xg, xl)

            test_outputs.append(outputs.cpu())

            probs = torch.sigmoid(outputs)

            predicted_classes = (probs > THRESHOLD).int().cpu().numpy().flatten()

            test_predictions.extend(predicted_classes)

    test_predictions = np.array(test_predictions)

    test_acc = accuracy_score(test_y_true, test_predictions)
    test_f1 = f1_score(test_y_true, test_predictions)
    test_precision = precision_score(test_y_true, test_predictions)
    test_recall = recall_score(test_y_true, test_predictions)

    targets = test_y.float().unsqueeze(1)

    test_outputs = torch.concatenate(test_outputs, dim = 0)

    test_loss = loss_function(test_outputs, targets)
        
    return test_acc, test_loss, test_f1, test_precision, test_recall
    

def cnn_eval(b_size, cnn_ev, device_ev, val_X_g_ev, val_X_l_ev, val_y_true_ev):

    cnn_ev.eval()

    predictions = []
    total_probs = []

    with torch.no_grad():
        for i in range(0,len(val_X_g_ev), b_size):

            xg = val_X_g_ev[i:i+b_size].to(device_ev)
            xl = val_X_l_ev[i:i+b_size].to(device_ev)

            outputs = cnn_ev(xg, xl)

            probs = torch.sigmoid(outputs) 

            total_probs.append(probs)

            predicted_classes = (probs > THRESHOLD).int().cpu().numpy().flatten()

            predictions.extend(predicted_classes)

    total_probs = torch.cat(total_probs, dim=0)

    predictions = np.array(predictions)

    accuracy = accuracy_score(val_y_true_ev, predictions)
    recall = recall_score(val_y_true_ev, predictions)
    precision = precision_score(val_y_true_ev, predictions)
    f1 = f1_score(val_y_true_ev, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")

    with open("data/ExoVetNetExperimentLog.csv", "a") as f:
        f.write(f"{MODEL_NAME},{accuracy},{recall},{precision},{f1},{EPOCHS},{BATCH_SIZE},{THRESHOLD},{NOISE_STD}\n")

    return f1, predictions, total_probs


def optimal_threshold(probs, val_y_true_t):

    best_t = 0
    best_f1 = 0

    ts = []
    f1s = []
    rs = []
    ps = []

    for t in tqdm(np.arange(0, 1.01, 0.01)):

        preds = (probs > t).cpu().numpy().astype(int)

        f1 = f1_score(val_y_true_t, preds)
        recall = recall_score(val_y_true_t, preds)
        precision = precision_score(val_y_true_t, preds)

        ts.append(t)
        f1s.append(f1)
        rs.append(recall)
        ps.append(precision)

        #print(f"Threshold: {t}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"Best Threshold {best_t}")
    print(f"Best F1 {best_f1}")

    style.use("ggplot")

    plt.plot(ts, f1s, label="F1")
    plt.plot(ts, rs, label = "Recall")
    plt.plot(ts, ps, label = "Precision")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance Scores vs Thresholds")
    plt.savefig(f"figures/optimal_thresholds_graph{time.time()}.png")

    plt.show()


    return best_t


def early_stopping_val(b_size, cnn_stop_val, val_X_g_stop_val, val_X_l_stop_val, val_y_stop_val):

    cnn_stop_val.eval()

    total_outputs = []

    with torch.no_grad():
        for i in range(0,len(val_X_g_stop_val), b_size):

            xg = val_X_g_stop_val[i:i+b_size].to(device)
            xl = val_X_l_stop_val[i:i+b_size].to(device)

            outputs = cnn_stop_val(xg, xl)

            total_outputs.append(outputs.cpu())


    total_outputs = torch.cat(total_outputs, dim = 0)
    targets = val_y_stop_val.float().unsqueeze(1)

    loss = loss_function(total_outputs, targets)

    return loss


def early_stopping(b_size, cnn_stop, optimizer_stop, train_X_g_stop, train_X_l_stop, train_y_stop, val_X_g_stop, val_X_l_stop, val_y_stop):

    best_loss = float('inf')
    best_state = None
    epochs_since_improvement = 0

    MAX_EPOCHS = 100
    PATIENCE = 20

    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCHS):
        for i in tqdm(range(0, len(train_X_g_stop), b_size)):

            cnn_stop.train()

            batch_X_g = train_X_g_stop[i:i+b_size].to(device)
            batch_X_l = train_X_l_stop[i:i+b_size].to(device)

            batch_X_g = batch_X_g + torch.randn_like(batch_X_g) * NOISE_STD
            batch_X_l = batch_X_l + torch.randn_like(batch_X_l) * NOISE_STD

            batch_y = train_y_stop[i:i+b_size].to(device)

            _, train_loss = fwd_pass(batch_X_g, batch_X_l, batch_y, optimizer_stop, cnn_stop, train=True)

            train_losses.append(train_loss)
            
        val_loss = early_stopping_val(b_size, cnn_stop, val_X_g_stop, val_X_l_stop, val_y_stop)
        val_losses.append(val_loss)

        print(f"epoch {epoch}: val_loss {float(val_loss):.4f}  (best {float(best_loss):.4f}, patience {epochs_since_improvement})")

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(cnn_stop.state_dict())
            epochs_since_improvement = 0

        else:
            epochs_since_improvement +=1

            if epochs_since_improvement >= PATIENCE:
                break

    print(f"best epoch: {best_epoch}, val_loss {float(best_loss):.4f}")

    cnn_stop.load_state_dict(best_state)
    return cnn_stop


def visualize_acc_loss(model_name = MODEL_NAME):

    style.use("ggplot")

    log_df = pd.read_csv("data/ExoVetNet_log.csv")
    log_df = log_df[log_df["model_name"] == model_name]

    log_df["timestamp"] = log_df["timestamp"].astype(float)
    log_df["epoch"] = log_df["epoch"].astype(float)
    log_df["step"] = log_df["step"].astype(float)

    log_df["accuracy"] = log_df["accuracy"].astype(float)
    log_df["validation_accuracy"] = log_df["validation_accuracy"].astype(float)

    log_df["loss"] = log_df["loss"].astype(float)
    log_df["validation_loss"] = log_df["validation_loss"].astype(float)

    # Plotting

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0),sharex=ax1)

    ax1.plot(log_df["step"], log_df["accuracy"], label = "acc")
    ax1.plot(log_df["step"],log_df["validation_accuracy"], label = "val_acc")
    ax1.legend(loc=2)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy of ExoVetNet over time")

    ax2.plot(log_df["step"], log_df["loss"] , label = "loss")
    ax2.plot(log_df["step"], log_df["validation_loss"] , label = "val_loss")
    ax2.legend(loc=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss of ExoVetNet over time")

    plt.show()


def plot_cm(cm):

    style.use("ggplot")

    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
    markers = [f"{count}\n{pct}" for count, pct in zip(group_counts, group_percentages)]
    markers = np.asarray(markers).reshape(2,2)

    plt.figure(figsize = (7,6), dpi=100)

    ax = sns.heatmap(
        cm, 
        annot=markers, 
        fmt='', 
        cmap='coolwarm', 
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='white',
        annot_kws={'fontsize': 12}
    )

    ax.set_title("ExoVetNet Confusion Matrix\n", fontsize = 14)
    ax.set_xlabel("Predicted", fontsize = 12, labelpad = 10)
    ax.set_ylabel("Actual", fontsize = 12, labelpad = 10)

    ax.xaxis.set_ticklabels(["False Positive", "Confirmed"], fontsize = 10)
    ax.yaxis.set_ticklabels(["False Positive", "Confirmed"], fontsize = 10, va = "center")

    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix{time.time()}.png")
    plt.show()


if __name__ == "__main__":


    # == Main ==


    # Assigning device (GPU is the fastest option)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")

    else:
        device = torch.device("cpu")
        print("Running on CPU")

    '''
    df = pd.read_csv('data/BLSvCatalog_data.csv')
    df['period_diff'] = np.abs(df['cat_period'] - df['period']) / df['cat_period']
    good_names = df[df['period_diff'] < 0.01]['name'].tolist()
    good_names = [n for n in good_names if os.path.exists(f"data/fluxes_bls/{n}.npz")]
    '''

    # importing flux arrays
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
    # Creating CNN

    cnn = Net().to(device)

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

    train_X_g = X_g[:-train_size]
    train_X_l = X_l[:-train_size]
    train_y = y[:-train_size]

    val_X_g = X_g[-train_size:-test_size]
    val_X_l = X_l[-train_size:-test_size]
    val_y = y[-train_size:-test_size]

    test_X_g = X_g[-test_size:]
    test_X_l = X_l[-test_size:]
    test_y = y[-test_size:]

    test_names = names[-test_size:]

    val_y_true = val_y.cpu().numpy()
    test_y_true = test_y.cpu().numpy()


    # K-Fold CV

    cv_X_g = torch.cat([train_X_g, val_X_g], dim = 0)
    cv_X_l = torch.cat([train_X_l, val_X_l], dim = 0)
    cv_y = torch.cat([train_y, val_y], dim = 0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    #thresholds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_X_g)):

        train_X_g_fold = cv_X_g[train_idx]
        val_X_g_fold = cv_X_g[val_idx]

        train_X_l_fold = cv_X_l[train_idx]
        val_X_l_fold = cv_X_l[val_idx]

        train_y_fold = cv_y[train_idx]
        val_y_fold = cv_y[val_idx]

        val_y_true_fold = val_y_fold.cpu().numpy()

        cnn = Net().to(device)
        optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

        cnn = early_stopping(BATCH_SIZE, cnn, optimizer, train_X_g_fold, train_X_l_fold, train_y_fold, val_X_g_fold, val_X_l_fold, val_y_fold)

        f1, predictions, probs = cnn_eval(BATCH_SIZE, cnn, device, val_X_g_fold, val_X_l_fold, val_y_true_fold)

        #t = optimal_threshold(probs, val_y_true_fold)
        #thresholds.append(t)

        print(f"Fold: {fold+1} F1 {f1:.4f}")

        fold_scores.append(f1)

    print(f"CV F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    #print(f"Optimal Threshold: {np.mean(thresholds)}")

    '''
    cnn = Net().to(device)

    cnn.load_state_dict(torch.load("model/ExoVetNet.pt"))

    cnn.eval()
    '''

    test_acc, test_loss, test_f1, test_precision, test_recall = test(cnn, BATCH_SIZE)

    print(f"Final Test Accuracy: {test_acc}")
    print(f"Final Test Loss: {test_loss}")

    print(f"Final Test F1: {test_f1}")
    print(f"Final Test Precision: {test_precision}")
    print(f"Final Test Recall: {test_recall}")

    f1, predictions, probs = cnn_eval(BATCH_SIZE, cnn, device, test_X_g, test_X_l, test_y_true)

    cm = confusion_matrix(test_y_true, y_pred = predictions) 
    plot_cm(cm)

    optimal_threshold(probs, test_y_true)

    missed_planets = [test_names[i] for i in range(len(test_y_true))
                    if test_y_true[i]==1 and predictions[i]==0]
    false_alarms  = [test_names[i] for i in range(len(test_y_true))
                    if test_y_true[i]==0 and predictions[i]==1] 

    feat = pd.read_csv("data/exoplanet_features.csv")
    missed = feat[feat["name"].isin(missed_planets)]


    missed.to_csv("data/missed_planets.csv", index=False)

    torch.save(cnn.state_dict(), "model/ExoVetNet.pt")

    missed = feat[feat["name"].isin(missed_planets)]
    all_planets = feat[feat["label"]==1]

