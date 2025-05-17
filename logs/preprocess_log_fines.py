import pandas as pd
import math
from collections import Counter
import numpy as np

def check_if_activity_exists(group, activity):
    label = 0
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    if relevant_activity_idxs.size > 0:
        idx = relevant_activity_idxs[0]
        group["label"] = 1
        group = group[:idx]
    else:
        group["label"] = 0
        group = group

    first_non_nan_amount = group["amount"].to_list()[0]
    relevant_activity_idxs = np.where(group["concept:name"] == "Payment")[0]
    if relevant_activity_idxs.size > 0:
        payment = group["paymentAmount"].to_list()[relevant_activity_idxs[0]]
        if (payment < first_non_nan_amount):
            group["rule_1"] = 1
        else:
            group["rule_1"] = 0
    else:
        group["rule_1"] = 0
    relevant_activity_idxs = np.where(group["concept:name"] == "Add penalty")[0]
    if group["points"].to_list()[0] >= 4 and relevant_activity_idxs.size > 0:
        group["rule_2"] = 1
    else:
        group["rule_2"] = 0
    relevant_activity_idxs = np.where(group["concept:name"] == "Add penalty")[0]
    if first_non_nan_amount > 400:
        group["rule_3"] = 1
    else:
        group["rule_3"] = 0
    return group

data = pd.read_csv("traffic_fines.csv")
data = data.dropna(subset=["case:concept:name"])

print(data.columns)


data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists, "Send for Credit Collection")
data_labeled.to_csv("traffic_fines_outcome.csv", index=False)

list_1 = data_labeled[data_labeled["concept:name"] == "Create Fine"]["rule_1"].to_list()
counts = Counter(list_1)

list_2 = data_labeled[data_labeled["concept:name"] == "Create Fine"]["rule_2"].to_list()
counts2 = Counter(list_2)

list_3 = data_labeled[data_labeled["concept:name"] == "Create Fine"]["rule_3"].to_list()
counts3 = Counter(list_3)


print(counts)
print(counts2)
print(counts3)