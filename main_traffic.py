import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA, LSTMModelNext, LogitsToPredicate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_accuracy, compute_metrics, compute_accuracy_a, compute_metrics_fa
from collections import defaultdict, Counter
from data import preprocess_sepsis
from data import preprocess_bpi12
from data import preprocess_traffic
from data.dataset import NeSyDataset, ModelConfig
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "traffic_fines"

classes = ["Repaid", "Send for credit collection"]

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
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=2, help="Number of epochs for training LTN model")
    # training configuration
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")

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

(X_train_, y_train_, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_traffic.preprocess_eventlog(data)

X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.2, stratify=y_train_, random_state=42)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
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
compliance_lstm = 0
num_constraints = 0
    
rule_penalty = lambda x: (x[:, :10] == 1).any(dim=1)
rule_payment = lambda x: (x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])
rule_amount = lambda x: (x[:, 90] > scalers["amount"].transform([[400]])[0][0])
count_r2 = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        # Apply the rule to the input data
        rule_penalty_res = rule_penalty(x).detach().cpu().numpy()
        rule_payment_res = rule_payment(x).detach().cpu().numpy()
        rule_amount_res = rule_amount(x).detach().cpu().numpy()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if rule_penalty_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
                    compliance_lstm += 1
            if rule_payment_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
                    compliance_lstm += 1
            if rule_amount_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
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
print(count_r2)
print(num_constraints)
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

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

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
        if x_P.value.numel() > 0:
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
    # print(" epoch %d | loss %.4f | Train Acc %.3f "
    #             %(epoch, train_loss, compute_accuracy_a(train_loader, lstm, device, rule_penalty, rule_payment, rule_amount)))
    print(" epoch %d | loss %.4f"
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :10] == 1).any(dim=1)),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, :10] == 7).any(dim=1) & (x.value[:, 100:110].max(dim=1).values < x.value[:, 90]))),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 90] > scalers["amount"].transform([[400]])[0][0])),
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
        rule_penalty_res = rule_penalty(x).detach()
        rule_payment_res = rule_payment(x).detach()
        rule_amount_res = rule_amount(x).detach()
        x = torch.cat([x, rule_penalty_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_payment_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_amount_res.unsqueeze(1).repeat(1, 10)], dim=1)
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
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w knowledge (A)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
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
        rule_penalty_res = rule_penalty(x).detach()
        rule_payment_res = rule_payment(x).detach()
        rule_amount_res = rule_amount(x).detach()
        x = torch.cat([x, rule_penalty_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_payment_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_amount_res.unsqueeze(1).repeat(1, 10)], dim=1)
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :10] == 1).any(dim=1)),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, :10] == 7).any(dim=1) & (x.value[:, 100:110].max(dim=1).values < x.value[:, 90]))),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 90] > scalers["amount"].transform([[400]])[0][0])),
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
print("Metrics LTN w knowledge (AB)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
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
lstm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
lstm_imm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
#has_act = ltn.Function(func= lambda x: torch.tensor(x[:, 104:117] == 5).any(dim=1))
has_act_1 = ltn.Function(func = lambda x: torch.tensor(x[:, :10] == 3).any(dim=1))
has_act_2 = ltn.Function(func = lambda x: torch.tensor(x[:, :10] == 10).any(dim=1))
SendFine = ltn.Constant(torch.tensor([1, 0, 0]))
Notification = ltn.Constant(torch.tensor([0, 1, 0]))
Payment = ltn.Constant(torch.tensor([0, 0, 1]))
P = ltn.Predicate(lstm).to(device)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)
ImmediateNext = ltn.Predicate(LogitsToPredicate(lstm_imm_next)).to(device)
SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters()) + list(Next.parameters()) + list(ImmediateNext.parameters())
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :10] == 1).any(dim=1)),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, :10] == 7).any(dim=1) & (x.value[:, 100:110].max(dim=1).values < x.value[:, 90]))),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 90] > scalers["amount"].transform([[400]])[0][0])),
            Forall(x_All, And(has_act_1(x_All), ImmediateNext(x_All, SendFine))),
            Forall(x_All, And(has_act_2(x_All), ImmediateNext(x_All, Notification))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, Payment))),
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
lstm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)
lstm_imm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
ImmediateNext = ltn.Predicate(LogitsToPredicate(lstm_imm_next)).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters()) + list(Next.parameters()) + list(ImmediateNext.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
# for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x_Next = ltn.Variable("x_All", x)
        rule_penalty_res = rule_penalty(x).detach()
        rule_payment_res = rule_payment(x).detach()
        rule_amount_res = rule_amount(x).detach()
        x = torch.cat([x, rule_penalty_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_payment_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_amount_res.unsqueeze(1).repeat(1, 10)], dim=1)
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
        formulas.extend([
            Forall(x_All, And(has_act_1(x_All), ImmediateNext(x_All, SendFine))),
            Forall(x_All, And(has_act_2(x_All), ImmediateNext(x_All, Notification))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, Payment))),
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
print("Metrics LTN w knowledge (AC)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
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
lstm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
Next = ltn.Predicate(LogitsToPredicate(lstm_next)).to(device)
lstm_imm_next = LSTMModelNext(vocab_sizes, config, 3, feature_names)
ImmediateNext = ltn.Predicate(LogitsToPredicate(lstm_imm_next)).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters()) + list(Next.parameters()) + list(ImmediateNext.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
#for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x_Next = ltn.Variable("x_All", x)
        rule_penalty_res = rule_penalty(x).detach()
        rule_payment_res = rule_payment(x).detach()
        rule_amount_res = rule_amount(x).detach()
        x = torch.cat([x, rule_penalty_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_payment_res.unsqueeze(1).repeat(1, 10)], dim=1)
        x = torch.cat([x, rule_amount_res.unsqueeze(1).repeat(1, 10)], dim=1)
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :10] == 1).any(dim=1)),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, :10] == 7).any(dim=1) & (x.value[:, 100:110].max(dim=1).values < x.value[:, 90]))),
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 90] > scalers["amount"].transform([[400]])[0][0])),
            Forall(x_All, And(has_act_1(x_All), ImmediateNext(x_All, SendFine))),
            Forall(x_All, And(has_act_2(x_All), ImmediateNext(x_All, Notification))),
            Forall(x_All, And(has_act_2(x_All), Next(x_All, Payment))),
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
print("Metrics LTN w knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
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