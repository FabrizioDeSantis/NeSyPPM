import pandas as pd
import numpy as np
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
from operator import itemgetter

data = pd.read_csv("bpi17.csv", dtype={"org:resource": str})
data = data.dropna(subset=["case:concept:name"])

def check_if_activity_exists(group, activity):
    req_amount = group["case:RequestedAmount"].to_list()[0]
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    label = 0
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group["label"] = 1
        label = 1
        group = group[:idx]
    else:
        relevant_activity_idxs = np.where(group["concept:name"] == "O_Cancelled")[0]
        if len(relevant_activity_idxs) > 0:
            idx = relevant_activity_idxs[0]
            group["label"] = 0
            group = group[:idx]
        else:
            relevant_activity_idxs = np.where(group["concept:name"] == "O_Refused")[0]
            if len(relevant_activity_idxs) > 0:
                idx = relevant_activity_idxs[0]
                group["label"] = 0
                group = group[:idx]
    # RULE 1
    relevant_activity_idxs = np.where(group["concept:name"] == "O_Create Offer")[0]
    if len(relevant_activity_idxs) > 0:
        credit_score = group["CreditScore"].to_list()[relevant_activity_idxs[0]]
        if credit_score > 0 and req_amount < 20000:
            group["rule_1"] = 1
        else:
            group["rule_1"] = 0
    relevant_activity_idxs = np.where(group["concept:name"] == "O_Create Offer")[0]
    # RULE 2
    if len(relevant_activity_idxs) > 0:
        credit_score = group["CreditScore"].to_list()[relevant_activity_idxs[0]]
        if credit_score == 0 and label == 0:
            group["rule_2"] = 1
        else:
            group["rule_2"] = 0
    # RULE 3
    
    loan_goal = group["case:LoanGoal"].to_list()[0]
    if req_amount > 20000 and loan_goal == "Existing loan takeover":
        group["rule_3"] = 1
    else:
        group["rule_3"] = 0
    return group


data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists, "O_Accepted")
data_labeled = data_labeled.dropna(subset=["label"])
data_labeled["label"] = data_labeled["label"].astype(int)
data_labeled.to_csv("bpi17_outcome.csv", index=False)

list_1 = data_labeled[data_labeled["concept:name"] == "A_Create Application"]["label"].to_list()
counts1 = Counter(list_1)
print(counts1)

list_2 = data_labeled[data_labeled["concept:name"] == "A_Create Application"]["rule_1"].to_list()
counts2 = Counter(list_2)
print(counts2)

list_3 = data_labeled[data_labeled["concept:name"] == "A_Create Application"]["rule_2"].to_list()
counts3 = Counter(list_3)
print(counts3)

list_4 = data_labeled[data_labeled["concept:name"] == "A_Create Application"]["rule_3"].to_list()
counts4 = Counter(list_4)
print(counts4)