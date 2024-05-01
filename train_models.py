import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import math
import shap
from os.path import join
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from itertools import product
import pickle
from scipy.stats import stats
from sklearn import metrics
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time

from tqdm import tqdm


# A method for deleting given columns safely
def safe_drop(data, columns=[], colrange=()):
    if colrange:
        if colrange[0] in data.columns and colrange[1] in data.columns:
            original_cols = list(data.columns.values)
            columns += [original_cols[i] for i in
                        range(original_cols.index(colrange[0]), original_cols.index(colrange[1]) + 1)]
    for col in columns:
        if col in data.columns:
            data.drop(col, inplace=True, axis=1)


# 1 Read the data

def process_data():
    data = pd.read_csv(
        'all_inputs.csv', skiprows=1)
    # Filling null values with median
    data.fillna(data.median(), inplace=True)
    # hospitalization parametes: "Fever_Min_during_hospitalization", "pulse_Min_during_hospitalization", "Saturation_Min_during_hospitalization", "Diastolic _BP_Min_during_hospitalization", "Systolic _BP_Min_during_hospitalization", "Cervical_length_Min_during_hospitalization", "Fever_Max_during_hospitalization", "pulse_max_during_hospitalization", "Saturation_Max_during_hospitalization", "Diastolic _BP_Max_during_hospitalization", "Systolic _BP_Max_during_hospitalization", "Cervical_length_Max_during_hospitalization", "Cervical_dilation_Max_during_hospitalization", "Cervical_effacement_Max_during_hospitalization", "Days with fever above 38", "Days of pulse above 100", "WBC_Min", "HB_Min", "PLT_Min", "ALK.PHOSPHATASE_Min", "AST_Min", "ALT_Min", "C-REACTIVE PROTEIN_Min", "MCV_Min", "RDW_Min", "Total Bile Acids_Min", "Transferrin_Min", "Albumin_Min", "Calcium_Min", "Protein-total_Min", "Phosphorus_Min", "TSH_Min", "T3-FREE_Min", "T4-FREE_Min", "Protein-Urine 24h_Min", "Creatinine- U sample_Min", "Protein - U sample_Min", "WBC_Max", "HB_Max", "PLT_Max", "ALK.PHOSPHATASE_Max", "AST_Max", "ALT_Max", "C-REACTIVE PROTEIN_Max", "CREATININE_Max", "MCV_Max", "RDW_Max", "GGT_Max", "Total Bile Acids_Max", "Transferrin_Max", "Ferritin_Max", "Albumin_Max", "Calcium_Max", "Protein-total_Max", "Phosphorus_Max", "TSH_Max", "T3-FREE_Max", "T4-FREE_Max", "Protein-Urine 24h_Max", "Creatinine- U sample_Max", "Protein - U sample_Max"
    return data


# 2. Function to split data into Train and Test.
def split_data(data, test_size=0.2, days=7):
    if days == 2:
        labels = data["Delivered within 2 days from admission"]
    elif days == 7:
        labels = data["Delivered within 7 days from admission"]
    elif days == 37:
        labels = data["Delivered before 37 yes no"]
    elif days == 34:
        labels = data["Delivered before 34 yes no"]
    else:
        print("Wrong threshold for labels.")
        return
    safe_drop(data, columns=["Delivered within 2 days from admission", "Delivered within 7 days from admission"
        , "Days from admission to delivery", "Delivered before 37 yes no", "Delivered before 34 yes no"])
    for col in data.columns:
        if (data[col].std(ddof=0) > 0):
            data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
    return train_test_split(data, labels, test_size=test_size, random_state=42)


# 3. Function to return a different type of model according to given input.
# The models are: XGBoost, SVM, LogReg.
def get_model(x_train, y_train, type):
    if type == "logistic":
        model = LogisticRegression()
        model.fit(x_train, y_train)
    elif type == "xgboost":
        model = xgb.XGBClassifier(gamma=8, max_depth=4, subsample=0.92, eta=0.34)
        model.fit(x_train, y_train)
    elif type == "fnn":
        train_auc, model = Learn_FFN(x_train, y_train)
    else:
        print("Model type parameter isn't valid.")
        return
    return model


# 4. Plot ROC-AUC Curve for the trained model.
def plot_auc(model, x_test, y_test, ax, type):
    # define metrics
    font_size = 16
    if type == "logistic" or type == "xgboost":
        y_pred = model.predict_proba(x_test)[:, 1]
    elif type == "svm":
        y_pred = model.decision_function(x_test)
    else:
        y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print("ROC score: ", roc_auc_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print("ROC score: ", roc_auc_score(y_test, y_pred))
    res_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds, })
    res_df.to_csv("Results/roc_curve_results_logistic_the entire cohort 7 days.csv")

    # Compute predictions
    y_pred = model.predict(x_test)

    # Calculate recall rate
    # recall = recall_score(y_test, y_pred)
    # print("Recall rate:", recall)

    # Calculate false positive rate
    false_positive_rate = np.mean(1 - tpr)

    # Calculate mean true positive rate
    true_positive_rate = np.mean(tpr)

    # Print false positive rate and true positive rate
    print("False Positive Rate:", false_positive_rate)
    print("Mean True Positive Rate:", true_positive_rate)

    # create ROC curve
    ax.plot(fpr, tpr)
    ax.set_ylabel('True Positive Rate', size=font_size)
    ax.set_xlabel('False Positive Rate', size=font_size)

    return y_pred, y_test  # Return predicted labels and true labels


# Smaller function just to get the AUC, I will plot it outside.
def get_auc(model, x_test, y_test, type):
    # define metrics
    font_size = 16
    if type == "logistic" or type == "xgboost":
        y_pred = model.predict_proba(x_test)[:, 1]
    elif type == "fnn":
        x_test = torch.tensor(x_test.values.astype(np.float32))
        #        y_test = torch.tensor(y_test.values.astype(np.float32))
        #        y_test = y_test.unsqueeze(1)
        y_pred = test_my_ffn(x_test, model)
    else:
        y_pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    precision, recall, thresholds1 = precision_recall_curve(y_test, y_pred)

    res_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds, })
    res_df.to_csv("Results/roc_curve_results_logistic_the entire cohort 7 days.csv")
    if type == "fnn":
        y_pred = y_pred.tolist()
    return auc, fpr, tpr, thresholds, precision, recall, thresholds1, y_pred, y_test


# Feature importance plot
def plot_importance(model, all_feature_importance, i):
    importance = model.get_booster().get_score(importance_type='gain')
    feature_list = []
    importance_score = []
    n_features = len(importance.items())
    # Print feature names and importance scores
    for feature, score in importance.items():
        print(f"{feature}: {score}")
        importance_score.append(score)
        feature_list.append(feature)
        if feature in all_feature_importance:
            vals = all_feature_importance[feature]
        else:
            vals = np.zeros(4)
        vals[i] = score
        all_feature_importance[feature] = vals  # export all vals to a dictionary
    return all_feature_importance


def plot_all(ax, i1, fpr, tpr, thresholds, precision, recall, thresholds1, y_pred, y_test, days=2):
    # Precision recall
    ax[0, i1].plot(precision, recall, label=type_model)
    ax[0, i1].set_xlabel('Precision', size=font_size)
    ax[0, i1].grid(True)  # Enable grid lines
    ax[0, i1].set_xlim(0, 1)  # Set x-axis limits
    ax[0, i1].set_ylim(0, 1)  # Set y-axis limits
    if (i1 < 2):
        title_text = f"Birth within {d1} days"  # f-string for formatted text
    else:
        title_text = f"Birth before week {d1}"  # f-string for formatted text

    ax[0, i1].set_title(title_text)
    # AUC
    ax[1, i1].plot(fpr, tpr, label=type_model)
    ax[1, i1].set_xlabel('FPR', size=font_size)
    ax[1, i1].grid(True)  # Enable grid lines
    ax[1, i1].set_xlim(0, 1)  # Set x-axis limits
    ax[1, i1].set_ylim(0, 1)  # Set y-axis limits

    # Positive rate vs recall
    all_test = {"Score": y_pred, "Tag": y_test}
    df = pd.DataFrame(all_test)
    df_sorted = df.sort_values(by="Score", ascending=False)
    df_sorted["Cumulative"] = df_sorted["Tag"].cumsum()
    z = df["Tag"].sum()
    df_sorted["Cumulative"] = df_sorted["Cumulative"]
    cumulative_sum = df_sorted["Cumulative"] / z

    # Generate x-axis values (0 to 1 with same length as cumulative_sum)
    x_axis = np.linspace(0, 1, len(cumulative_sum))  # Evenly spaced values from 0 to 1

    ax[2, i1].plot(x_axis, cumulative_sum, label=type_model)
    ax[2, i1].set_xlabel('Positve rate', size=font_size)
    ax[2, i1].grid(True)  # Enable grid lines
    ax[2, i1].set_xlim(0, 1)  # Set x-axis limits
    ax[2, i1].set_ylim(0, 1)  # Set y-axis limits

    # last row is risk, I compute it from both directions only for xgboost
    if (type_model == 'xgboost'):
        tot_samp = np.linspace(1, len(cumulative_sum), len(cumulative_sum))  # Evenly spaced values from 1 to b_Values
        ratios = [a / b * z for a, b in zip(cumulative_sum, tot_samp)]  # This is the risk from top to bottom

        df_sorted['reversed_Cumulative'] = df_sorted.loc[::-1, 'Tag'].cumsum()[::-1]
        inv_cumulative_sum = df_sorted["reversed_Cumulative"]
        inv_tot_samp = range(len(cumulative_sum), 0, -1)
        ratios_inv = [a / b for a, b in zip(inv_cumulative_sum, inv_tot_samp)]  # This is the risk from top to bottom

        ax[3, i1].plot(x_axis, ratios, label='Risk')
        ax[3, i1].plot(x_axis, ratios_inv, label='Inverted Risk')
        ax[3, i1].set_xlabel('Positve rate', size=font_size)
        ax[3, i1].grid(True)  # Enable grid lines
        ax[3, i1].set_xlim(0, 1)  # Set x-axis limits
        ax[3, i1].set_ylim(0, 1)  # Set y-axis limits
        ax[3, 0].legend()

    if i1 == 0:
        ax[0, i1].set_ylabel('Recall', size=font_size)
        ax[1, i1].set_ylabel('TPR', size=font_size)
        ax[2, i1].set_ylabel('Recall', size=font_size)
        ax[3, i1].set_ylabel('Risk', size=font_size)
        ax[0, i1].legend()


# The following part is to train my Fully Connected NN
class My_ffn(torch.nn.Module):
    def __init__(self, N_Inp, N_Size_1, N_Size_2, do_rate):
        super(My_ffn, self).__init__()
        self.fc1 = torch.nn.Linear(N_Inp, N_Size_1)
        self.fc2 = torch.nn.Linear(N_Size_1, N_Size_2)
        self.fc3 = torch.nn.Linear(N_Size_2, 1)
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        #    h_relu = self.linear1(x).clamp(min=0)
        #    y_pred = self.linear2(h_relu)
        #    return y_pred

        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.sigmoid(self.fc3(x))
        return x


def train_my_ffn(train_dataloader, model, loss_fn, optimizer):
    for x, y in train_dataloader:
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_my_ffn(x, model):
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return pred


def Learn_FFN(x_train, y_train, learn_rate=5e-2, N_S1=50, N_S2=20, weight_decay=0.01, do_rate=0.3, epochs=100):
    # Train the network and compute
    N, D_in = x_train.shape[0], x_train.shape[1]
    train_auc_all = np.zeros(epochs)
    test_auc_all = np.zeros(epochs)

    # Create random Tensors to hold inputs and outputs
    x_train = torch.tensor(x_train.values.astype(np.float32))
    y_train = torch.tensor(y_train.values.astype(np.float32))
    y_train = y_train.unsqueeze(1)

    # Construct our model by instantiating the class defined above.
    model = My_ffn(D_in, N_S1, N_S2, do_rate)

    # loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    training_set = TensorDataset(Tensor(x_train), Tensor(y_train))
    train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)

    for t in range(epochs):
        #        train_pred = train_my_ffn(X_train, y_train, model, loss_fn, optimizer)
        train_my_ffn(train_dataloader, model, loss_fn, optimizer)

        train_pred = test_my_ffn(x_train, model)
        train_auc = metrics.roc_auc_score(y_train, train_pred)

    return train_auc, model


fig, ax = plt.subplots(4, 4, figsize=(15, 10))

data_shape = (4, 3)
font_size = 12
AUCs = np.zeros(data_shape)
all_feature_importance = {}

# Iterate over the possible days
for i, d1 in tqdm(enumerate([7, 2, 34]), desc="Days:"):
    alldata = process_data()
    x_train, x_test, y_train, y_test = split_data(alldata, test_size=0.2, days=d1)

    # Set the top 10 features
    top_features = ['Gestational age at admission', 'Parity', 'Maximal pulse at admission',
                    'Amniotic Fluid Index at admission',
                    'Cervical dynamics', 'Previous hospitalizations during pregnancy',
                    'Premature preterm rupture of membranes',
                    'Gestational hypertensive disorders', 'Cervical dilation', "Hemoglobin at admission"]

    # Filter x_train by the top features
    x_train = x_train[top_features]

    # Iterate over the model types
    for j, type_model in tqdm(enumerate(["logistic", "xgboost", "fnn"]), desc="Models:"):
        # Train the model
        model = get_model(x_train, y_train, type_model)
        # Save the model
        with open(join("app", "models", f"{type_model}_{d1}.pkl"), "wb") as f:
            pickle.dump(model, f)

#         # Calculate all sort of metrics
#         AUCs[i, j], fpr, tpr, thresholds, precision, recall, thresholds1, y_pred, y_test = get_auc(model, x_test, y_test,
#                                                                                                    type_model)
#         # Calculate and plot the FI
#         all_feature_importance = plot_importance(model, all_feature_importance, i)
#         # Plot the metrics
#         plot_all(ax, i, fpr, tpr, thresholds, precision, recall, thresholds1, y_pred, y_test, days=d1)
#
# # Save the figure
# plt.savefig('Results/accuracy.png', bbox_inches='tight')
