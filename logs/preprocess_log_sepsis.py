import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

def check_if_activity_exists(group, activity):
    boolean_ok = False
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group["label"] = 1
        group = group[:idx]
        boolean_ok = True
    else:
        relevant_activity_idxs = np.where(group["concept:name"] == "Release A")[0]
        if len(relevant_activity_idxs) > 0:
            idx = relevant_activity_idxs[0]
            group["label"] = 0
            group = group[:idx]
            boolean_ok = True
        else:
            relevant_activity_idxs = np.where(group["concept:name"] == "Release B")[0]
            if len(relevant_activity_idxs) > 0:
                idx = relevant_activity_idxs[0]
                group["label"] = 0
                group = group[:idx]
                boolean_ok = True
            else:
                relevant_activity_idxs = np.where(group["concept:name"] == "Release C")[0]
                if len(relevant_activity_idxs) > 0:
                    idx = relevant_activity_idxs[0]
                    group["label"] = 0
                    group = group[:idx]
                    boolean_ok = True
                else:
                    relevant_activity_idxs = np.where(group["concept:name"] == "Release D")[0]
                    if len(relevant_activity_idxs) > 0:
                        idx = relevant_activity_idxs[0]
                        group["label"] = 0
                        group = group[:idx]
                        boolean_ok = True
                    else:
                        relevant_activity_idxs = np.where(group["concept:name"] == "Release E")[0]
                        if len(relevant_activity_idxs) > 0:
                            idx = relevant_activity_idxs[0]
                            group["label"] = 0
                            group = group[:idx]
                            boolean_ok = True
    if boolean_ok:
        # RULE 1: ALL LACTIC ACID VALUES > 2
        indexes = group[group["concept:name"] == "LacticAcid"].index
        if len(indexes) > 0:
            values = group.loc[indexes, "LacticAcid"].to_list()
            if all(value > 2 for value in values):
                group["rule_1"] = 1
            else:
                group["rule_1"] = 0
        else:
            group["rule_1"] = 0
        #####################################
        # RULE 2: SIRSCritTachypnea AND InfectionSuspected AND SIRSCritHeartRate
        indexes = group[group["concept:name"] == "ER Registration"].index
        if len(indexes) > 0:
            tachypnea = group.loc[indexes, "SIRSCritTachypnea"].to_list()[0]
            infection_suspected = group.loc[indexes, "InfectionSuspected"].to_list()[0]
            heart_rate = group.loc[indexes, "SIRSCritHeartRate"].to_list()[0]
            if tachypnea == 1 and infection_suspected == 1 and heart_rate == 1:
                group["rule_2"] = 1
            else:
                group["rule_2"] = 0
        else:
            group["rule_2"] = 0
        #####################################
        # RULE 3: CRP followed by IV Antibiotics and CRP value > 100
        indexes = group[group["concept:name"] == "CRP"].index
        indexes_antibiotics = group[group["concept:name"] == "IV Antibiotics"].index
        found = False
        if len(indexes) > 0 and len(indexes_antibiotics) > 0:
            for index in indexes:
                for index_at in indexes_antibiotics:
                    if not found:
                        if index_at > index:
                            crp_value = group.loc[index, "CRP"].tolist()
                            if crp_value > 100:
                                group["rule_3"] = 1
                                found = True
            if not found:
                group["rule_3"] = 0
        else:
            group["rule_3"] = 0
        return group
    # else:
    #     group["label"] = 0
    #     group["rule_1"] = 0
    #     group["rule_2"] = 0
    #     group["rule_3"] = 0
    #     return group
        
def check_if_activity_exists_2(group, activity):
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group["label"] = 0
        return group[:idx]
    else:
        relevant_activity_idxs = np.where(group["concept:name"] == "Release B")[0]
        if len(relevant_activity_idxs) > 0:
            idx = relevant_activity_idxs[0]
            group["label"] = 1
            return group[:idx]
        else:
            relevant_activity_idxs = np.where(group["concept:name"] == "Release C")[0]
            if len(relevant_activity_idxs) > 0:
                idx = relevant_activity_idxs[0]
                group["label"] = 1
                return group[:idx]
            else:
                relevant_activity_idxs = np.where(group["concept:name"] == "Release D")[0]
                if len(relevant_activity_idxs) > 0:
                    idx = relevant_activity_idxs[0]
                    group["label"] = 1
                    return group[:idx]
                else:
                    relevant_activity_idxs = np.where(group["concept:name"] == "Release E")[0]
                    if len(relevant_activity_idxs) > 0:
                        idx = relevant_activity_idxs[0]
                        group["label"] = 1
                        return group[:idx]
                    else:
                        group["label"] = 1
                        return group
                    
def check_if_any_of_activities_exist(group, activities):
    if np.sum(group["concept:name"].isin(activities)) > 0:
        return True
    else:
        return False
    
timestamp_col = "time:timestamp"
def extract_timestamp_features(group):
    
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    # group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes
    group["timesincelastevent"] = tmp
    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(0)
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)
    
    return group

def check_if_activity_exists_and_time_less_than(group, activity):
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        if group["timesincelastevent"].iloc[idx] <= 28 * 1440: # return in less than 28 days
            group["label"] = 1
            return group[:idx]
        else:
            group["label"] = 0
            return group[:idx]
    else:
        group["label"] = 0
        return group
    
data = pd.read_csv("sepsis.csv")
data = data.dropna(subset=["case:concept:name"])

tmp = data.groupby("case:concept:name").apply(check_if_any_of_activities_exist, activities=["Release A", "Release B", "Release C", "Release D", "Release E"])
incomplete_cases = tmp.index[tmp==False]
data = data[~data["case:concept:name"].isin(incomplete_cases)]

# data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists, "Admission IC")
# data_labeled.to_csv("sepsis_2.csv", index=False)

# data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists_2, "Release A")
# data_labeled.to_csv("sepsis_3.csv", index=False)

data_labeled = data.groupby("case:concept:name").apply(check_if_activity_exists, "IV Antibiotics")
data_labeled.to_csv("sepsis_1.csv", index=False)
print(Counter(data_labeled["label"][data_labeled["concept:name"] == "ER Triage"]))
