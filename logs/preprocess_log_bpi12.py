import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("bpi12.csv", dtype={"org:resource": str})
data = data.dropna(subset=["case:concept:name"])
data["concept:name"] = data["concept:name"] + "-" + data["lifecycle:transition"]
data = data.drop(columns=["lifecycle:transition"])

def check_if_activity_exists(group):
    relevant_activity_idxs = np.where(group["concept:name"] == "O_ACCEPTED-COMPLETE")[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        req_amount = group["case:AMOUNT_REQ"].tolist()[0]
        if req_amount > 0 and req_amount < 10000:
            group["rule_1"] = 1
        else:
            group["rule_1"] = 0
        if req_amount > 40000 and req_amount <= 60000:
            group["rule_2"] = 1
        else:
            group["rule_2"] = 0
        has_target_value = group["org:resource"].isin(["10910", "11169"]).any()
        if has_target_value:
            group["rule_3"] = 1
        else:
            group["rule_3"] = 0
        group["label"] = 1
        return group[:idx]
    else:
        relevant_activity_idxs = np.where(group["concept:name"] == "O_CANCELLED-COMPLETE")[0]
        if len(relevant_activity_idxs) > 0:
            idx = relevant_activity_idxs[0]
            req_amount = group["case:AMOUNT_REQ"].tolist()[0]
            if req_amount > 0 and req_amount < 10000:
                group["rule_1"] = 1
            else:
                group["rule_1"] = 0
            if req_amount > 40000 and req_amount <= 60000:
                group["rule_2"] = 1
            else:
                group["rule_2"] = 0
            has_target_value = group["org:resource"].isin(["10910", "11169"]).any()
            if has_target_value:
                group["rule_3"] = 1
            else:
                group["rule_3"] = 0
            group["label"] = 0
            return group[:idx]
        else:
            relevant_activity_idxs = np.where(group["concept:name"] == "O_DECLINED-COMPLETE")[0]
            if len(relevant_activity_idxs) > 0:
                idx = relevant_activity_idxs[0]
                req_amount = group["case:AMOUNT_REQ"].tolist()[0]
                if req_amount > 40000 and req_amount <= 60000:
                    group["rule_2"] = 1
                else:
                    group["rule_2"] = 0
                if req_amount > 0 and req_amount < 10000:
                    group["rule_1"] = 1
                else:
                    group["rule_1"] = 0
                has_target_value = group["org:resource"].isin(["10910", "11169"]).any()
                if has_target_value:
                    group["rule_3"] = 1
                else:
                    group["rule_3"] = 0
                group["label"] = 0
                return group[:idx]
            
data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists)

data_labeled.to_csv("bpi12_outcome.csv", index=False)