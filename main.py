import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA, LSTMModelNext, LogitsToPredicate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_accuracy, compute_metrics
from collections import defaultdict, Counter
from data import preprocess_sepsis
from data import preprocess_bpi12
from data.dataset import NeSyDataset, ModelConfig, BPI12Dataset
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "sepsis_2"

if dataset == "sepsis_2":
    classes = ['No ICU', 'ICU']
elif dataset == "sepsis_3":
    classes = ["Release A", "No Release A"]
elif dataset == "bpi12":
    classes = ["Not accepted", "Accepted"]

metrics_lstm = []
metrics_ltn = []
metrics_ltn_A = []
metrics_ltn_B = []
metrics_ltn_AB = []
metrics_ltn_BC = []
metrics_ltn_AC = []
metrics_ltn_ABC = []

def get_args():
    parser = argparse.ArgumentParser()

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=5, help="Number of epochs for training LTN model")
    # training configuration
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--dataset_size", type=float, default=10, help="Size of the dataset (10%, 20%, 50%, 70%, 90%, 100%)")

    return parser.parse_args()

args = get_args()

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs = args.num_epochs
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("-- Reading dataset")
data = pd.read_csv("data_processed/"+dataset+".csv", dtype={"org:resource": str})

if dataset == "sepsis_2":
    (X_train_, y_train_, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_sepsis.preprocess_eventlog(data)
elif dataset == "bpi12":
    (X_train_, y_train_, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(data)

X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.2, stratify=y_train_, random_state=42)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

if dataset == "sepsis_2":
    train_dataset = NeSyDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = NeSyDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = NeSyDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
elif dataset == "bpi12":
    train_dataset = BPI12Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = BPI12Dataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = BPI12Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

lstm.train()
training_losses = []
validation_losses = []
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output.squeeze(1).cpu(), y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
    training_losses.append(statistics.mean(train_losses))
    lstm.eval()
    val_losses = []
    for enum, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x = x.to(device)
            output = lstm(x)
            loss = criterion(output.squeeze(1).cpu(), y)
            val_losses.append(loss.item())
    print(f"Validation Loss: {statistics.mean(val_losses)}")
    validation_losses.append(statistics.mean(val_losses))
    if epoch >= 5:
        if validation_losses[-1] > validation_losses[-2]:
            print("Validation loss increased, stopping training")
            break
    lstm.train()

lstm.eval()
y_pred = []
y_true = []
### SEPSIS RULES
rule_1 = lambda x: (x[..., 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1)
rule_2 = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device)
rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)
### BPI12 RULES
# rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
# rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
# rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
# rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
# rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)
compliance_lstm = 0
num_constraints = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        # Apply the rule to the input data
        if dataset == "sepsis_2":
            rule_2_res = rule_1(x).detach().cpu().numpy()
            rule_crp_atb_res = rule_crp_atb(x).detach().cpu().numpy()
            rule_crp_100_res = rule_crp_100(x).detach().cpu().numpy()
        elif dataset == "bpi12":
            rule_amount_1_res = rule_amount_1(x).detach().cpu().numpy()
            rule_amount_2_res = rule_amount_2(x).detach().cpu().numpy()
            rule_amount_3_res = rule_amount_3(x).detach().cpu().numpy()
            rule_resource_1_res = rule_resource_1(x).detach().cpu().numpy()
            rule_resource_2_res = rule_resource_2(x).detach().cpu().numpy()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if dataset == "sepsis_2":
                if rule_2_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        compliance_lstm += 1
                if rule_crp_atb_res[i] == 1 and rule_crp_100_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        compliance_lstm += 1
            elif dataset == "bpi12":
                if rule_amount_1_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        compliance_lstm += 1
                if rule_amount_2_res[i] == 1 and rule_amount_3_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        compliance_lstm += 1
                if rule_resource_1_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        compliance_lstm += 1
                if rule_resource_2_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        compliance_lstm += 1

print("Metrics LSTM")
accuracy = accuracy_score(y_true, y_pred)
metrics_lstm.append(accuracy)
print("Accuracy:", accuracy)
f1 = f1_score(y_true, y_pred, average='macro')
metrics_lstm.append(f1)
print("F1 Score:", f1)
precision = precision_score(y_true, y_pred, average='macro')
metrics_lstm.append(precision)
print("Precision:", precision)
recall = recall_score(y_true, y_pred, average='macro')
metrics_lstm.append(recall)
print("Recall:", recall)
print("Compliance:", compliance_lstm / num_constraints)
metrics_lstm.append(compliance_lstm / num_constraints)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_lstm.png", dpi=300, bbox_inches='tight')
plt.close()

lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)
#

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def compute_satisfaction_level(loader):
    mean_sat = 0
    for enum, (x, y) in enumerate(loader):
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        mean_sat += SatAgg(
            *formulas
        ).detach().cpu()
        del x_P, x_not_P
    mean_sat /= len(loader)
    return mean_sat

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f "
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w/o knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn.append(f1score)
print("Precision:", precision)
metrics_ltn.append(precision)
print("Recall:", recall)
metrics_ltn.append(recall)
print("Compliance:", compliance)
metrics_ltn.append(compliance)

# LTN_B

lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def compute_satisfaction_level(loader):
    mean_sat = 0
    for enum, (x, y) in enumerate(loader):
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        mean_sat += SatAgg(
            *formulas
        ).detach().cpu()
        del x_P, x_not_P
    mean_sat /= len(loader)
    return mean_sat

if dataset == "sepsis_2":
    f1 = ltn.Function(func=lambda x: (x[:, 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1))
    f2 = ltn.Function(func=lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1)))
    check_presence_crp_atb = ltn.Function(func= lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device))
    check_crp_100 = ltn.Function(func = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1))
    ERSepsisTriage = ltn.Constant(torch.tensor([1, 0]))
    Leucocytes = ltn.Constant(torch.tensor([0, 1]))
elif dataset == "bpi12":
    f1 = ltn.Function(func=lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1))
    f2 = ltn.Function(func=lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1))
    f3 = ltn.Function(func=lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))
    f_resources_11169 = ltn.Function(func=lambda x: (x[:, :240] == 48).any(dim=1))
    f_resources_10910 = ltn.Function(func=lambda x: (x[:, :240] == 21).any(dim=1))
    A_ACCEPTED_COMPLETE = ltn.Constant(torch.tensor([1, 0]))
    O_ACCEPTED_COMPLETE = ltn.Constant(torch.tensor([0, 1]))

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            # SEPSIS KG
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :13].eq(1).any(dim=1)) & (x.value[:, 39:52].eq(1).any(dim=1)) & (x.value[:, 65:78].eq(1).any(dim=1))),
            Forall(x_All, Implies(f2(x_All), P(x_All))),
            Forall(x_All, Implies(And(check_presence_crp_atb(x_All), check_crp_100(x_All)), P(x_All)))
            # # BPI 12 KG
            # Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            # Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            # Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            # Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w knowledge (B)")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_B.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_B.append(f1score)
print("Precision:", precision)
metrics_ltn_B.append(precision)
print("Recall:", recall)
metrics_ltn_B.append(recall)
print("Compliance:", compliance)
metrics_ltn_B.append(compliance)

# LTN_A

lstm = LSTMModelA(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f | Train Sat %.3f | Train Acc %.3f "
                %(epoch, train_loss, compute_satisfaction_level(train_loader), compute_accuracy(train_loader, lstm, device)))

lstm.eval()
print("Metrics LTN w knowledge (A)")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_A.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_A.append(f1score)
print("Precision:", precision)
metrics_ltn_A.append(precision)
print("Recall:", recall)
metrics_ltn_A.append(recall)
print("Compliance:", compliance)
metrics_ltn_A.append(compliance)

# LTN_AB

lstm = LSTMModelA(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P))),
            ])
        formulas.extend([
            # Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :13].eq(1).any(dim=1)) & (x.value[:, 39:52].eq(1).any(dim=1)) & (x.value[:, 65:78].eq(1).any(dim=1))),
            # Forall(x_All, Implies(f2(x_All), P(x_All))),
            # Forall(x_All, Implies(And(check_presence_crp_atb(x_All), check_crp_100(x_All)), P(x_All)))
            # BPI 12 KG
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f | Train Sat %.3f | Train Acc %.3f "
                %(epoch, train_loss, compute_satisfaction_level(train_loader), compute_accuracy(train_loader, lstm, device)))

lstm.eval()
print("Metrics LTN w knowledge (AB)")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_AB.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_AB.append(f1score)
print("Precision:", precision)
metrics_ltn_AB.append(precision)
print("Recall:", recall)
metrics_ltn_AB.append(recall)
print("Compliance:", compliance)
metrics_ltn_AB.append(compliance)

# LTN_BC

lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
lstm_next = LSTMModelNext(vocab_sizes, config, 2, feature_names)
#has_act = ltn.Function(func= lambda x: torch.tensor(x[:, 104:117] == 5).any(dim=1))
has_act_1 = ltn.Function(func = lambda x: torch.tensor(x[:, 40:80] == 22).any(dim=1))
has_act_2 = ltn.Function(func = lambda x: torch.tensor(x[:, 40:80] == 31).any(dim=1))
P = ltn.Predicate(lstm).to(device)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)
SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters()) + list(Next.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
# for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            # SEPSIS KG
            # Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :13].eq(1).any(dim=1)) & (x.value[:, 39:52].eq(1).any(dim=1)) & (x.value[:, 65:78].eq(1).any(dim=1))),
            # Forall(x_All, Implies(f2(x_All), P(x_All))),
            # Forall(x_All, Implies(And(check_presence_crp_atb(x_All), check_crp_100(x_All)), P(x_All))),
            # Forall(x_All, And(has_act(x_All), Next(x_All, ERSepsisTriage)))
            # BPI 12 KG
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            Forall(x_All, And(has_act_1(x_All), Next(x_All, A_ACCEPTED_COMPLETE))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, O_ACCEPTED_COMPLETE))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f | Train Sat %.3f | Train Acc %.3f "
                %(epoch, train_loss, compute_satisfaction_level(train_loader), compute_accuracy(train_loader, lstm, device)))

lstm.eval()
print("Metrics LTN w knowledge and parallel constraints")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_BC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_BC.append(f1score)
print("Precision:", precision)
metrics_ltn_BC.append(precision)
print("Recall:", recall)
metrics_ltn_BC.append(recall)
print("Compliance:", compliance)
metrics_ltn_BC.append(compliance)

# LTN_AC

lstm = LSTMModelA(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)
lstm_next = LSTMModelNext(vocab_sizes, config, 2, feature_names)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
# for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            # Forall(x_All, And(has_act(x_All), Next(x_All, ERSepsisTriage)))
            Forall(x_All, And(has_act_1(x_All), Next(x_All, A_ACCEPTED_COMPLETE))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, O_ACCEPTED_COMPLETE))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f | Train Sat %.3f | Train Acc %.3f "
                %(epoch, train_loss, compute_satisfaction_level(train_loader), compute_accuracy(train_loader, lstm, device)))

lstm.eval()
print("Metrics LTN w knowledge (AC)")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_AC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_AC.append(f1score)
print("Precision:", precision)
metrics_ltn_AC.append(precision)
print("Recall:", recall)
metrics_ltn_AC.append(recall)
print("Compliance:", compliance)
metrics_ltn_AC.append(compliance)

# LTN_ABC

lstm = LSTMModelA(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(lstm).to(device)
lstm_next = LSTMModelNext(vocab_sizes, config, 2, feature_names)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters()) + list(Next.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
#for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            # Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :13].eq(1).any(dim=1)) & (x.value[:, 39:52].eq(1).any(dim=1)) & (x.value[:, 65:78].eq(1).any(dim=1))),
            # Forall(x_All, Implies(f2(x_All), P(x_All))),
            # Forall(x_All, Implies(And(check_presence_crp_atb(x_All), check_crp_100(x_All)), P(x_All))),
            # Forall(x_All, And(has_act(x_All), Next(x_All, ERSepsisTriage))),
            # BPI 12 KG
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            Forall(x_All, And(has_act_1(x_All), Next(x_All, A_ACCEPTED_COMPLETE))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, O_ACCEPTED_COMPLETE))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f | Train Sat %.3f | Train Acc %.3f "
                %(epoch, train_loss, compute_satisfaction_level(train_loader), compute_accuracy(train_loader, lstm, device)))

lstm.eval()
print("Metrics LTN w knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_ABC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_ABC.append(f1score)
print("Precision:", precision)
metrics_ltn_ABC.append(precision)
print("Recall:", recall)
metrics_ltn_ABC.append(recall)
print("Compliance:", compliance)
metrics_ltn_ABC.append(compliance)

with open("metrics.txt", "a") as f:
    f.write("Accuracy, F1, Precision, Recall, Compliance\n")
    f.write("LSTM: \n")
    f.write(str(metrics_lstm))
    f.write("\n")
    f.write("LTN: \n")
    f.write(str(metrics_ltn))
    f.write("\n")
    f.write("LNT_B: \n")
    f.write(str(metrics_ltn_B))
    f.write("\n")
    f.write("LTN_A: \n")
    f.write(str(metrics_ltn_A))
    f.write("\n")
    f.write("LTN_AB: \n")
    f.write(str(metrics_ltn_AB))
    f.write("\n")
    f.write("LTN_BC: \n")
    f.write(str(metrics_ltn_BC))
    f.write("\n")
    f.write("LTN_AC: \n")
    f.write(str(metrics_ltn_AC))
    f.write("\n")
    f.write("LTN_ABC: \n")
    f.write(str(metrics_ltn_ABC))
    f.write("\n")