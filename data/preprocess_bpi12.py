from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data):
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())  # Get the unique group IDs
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * 0.3)

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 0)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 0)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 0)]

    filtered_values_nor = [v for v, l, r1, r2, r3 in zip(unique_groups, labels, labels_r1, labels_r2, labels_r3) if (r1 == 0 and r2 == 0 and r3 == 0 and l == 1)]

    # filtered_values_r1_training = random.sample(filtered_values_r1, len(filtered_values_r1) // 2)
    # filtered_values_r1_test = [x for x in filtered_values_r1 if x not in filtered_values_r1_training]
    # print("Len compliant traces in training set: ", len(filtered_values_r1_training))
    # print("Len compliant traces in test set: ", len(filtered_values_r1_test))
    # filtered_values_r2_training = random.sample(filtered_values_r2, len(filtered_values_r2) // 2)
    # filtered_values_r2_test = [x for x in filtered_values_r2 if x not in filtered_values_r2_training]
    # print("Len compliant traces in training set: ", len(filtered_values_r2_training))
    # print("Len compliant traces in test set: ", len(filtered_values_r2_test))
    # filtered_values_r3_training = random.sample(filtered_values_r3, len(filtered_values_r3) // 2)
    # filtered_values_r3_test = [x for x in filtered_values_r3 if x not in filtered_values_r3_training]
    # print("Len compliant traces in training set: ", len(filtered_values_r3_training))
    # print("Len compliant traces in test set: ", len(filtered_values_r3))

    #print("Number of compliant rules in the training set: ", len(list(set(filtered_values_r1_training + filtered_values_r2_training + filtered_values_r3_training))))

    filtered_values = filtered_values_r1 + filtered_values_r2 + filtered_values_r3 + filtered_values_nor
    # filtered_values = filtered_values_r1_test + filtered_values_r2_test + filtered_values_r3_test + filtered_values_nor
    compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3))
    filtered_values = list(set(filtered_values))  # Remove duplicates

    print("Len filtered values: ", len(filtered_values))
    # take len_test_set from filtered_values
    if len(filtered_values) > len_test_set:
          # Set seed for reproducibility
        test_ids = random.sample(filtered_values, len_test_set)
    else:
        test_ids = filtered_values

    # Remove the selected test IDs from the original list
    training_ids = [x for x in unique_groups if x not in test_ids]
    #training_ids = training_ids + filtered_values_r1_training + filtered_values_r2_training + filtered_values_r3_training
    training_ids = list(set(training_ids))  # Remove duplicates

    compliant_training_ids = [x for x in training_ids if x in compliant_ids]
    compliant_test_ids = [x for x in test_ids if x in compliant_ids]
    print("Number of compliant traces in training set: ", len(compliant_training_ids))
    print("Number of compliant traces in test set: ", len(compliant_test_ids))

    return training_ids, test_ids

# def create_test_set(data):
#     random.seed(42)
#     grouped = data.groupby("case:concept:name")
#     unique_groups = list(grouped.groups.keys())  # Get the unique group IDs
#     labels = data.groupby("case:concept:name")["label"].first().to_list()

#     len_test_set = int(len(unique_groups) * 0.2)
#     len_training_set = len(unique_groups) - len_test_set

#     labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
#     labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
#     labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()

#     filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 0)]
#     filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 0)]
#     filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 0)]
#     compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3))

#     filtered_no_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l != 0)]
#     filtered_no_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l != 0)]
#     filtered_no_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l != 0)]
#     non_compliant_ids = list(set(filtered_no_r1 + filtered_no_r2 + filtered_no_r3))

#     print("Len training set: ", len_training_set)
#     print("Len test set: ", len_test_set)
#     print("Number of compliant traces: ", len(compliant_ids))
#     print("Number of non-compliant traces: ", len(non_compliant_ids))

#     if len(compliant_ids) > len_test_set:
#         test_ids = random.sample(compliant_ids, int(len_test_set*0.7))
#         # fill the rest with ids from unique_groups that are not in non_compliant_ids
#         remaining_ids = [x for x in unique_groups if x not in test_ids and x not in non_compliant_ids]
#         remaining_ids = random.sample(remaining_ids, len_test_set - len(test_ids))
#         test_ids = test_ids + remaining_ids
#     else:
#         test_ids = compliant_ids

#     training_ids = [x for x in unique_groups if x not in test_ids]

#     #### 
#     print("Compliant traces in training set: ", len([x for x in training_ids if x in compliant_ids]))
#     print("Compliant traces in test set: ", len([x for x in test_ids if x in compliant_ids]))

#     return training_ids, test_ids

def create_ngrams(data, train_ids, test_ids, window_size=40):

    ngrams_test = []
    ngrams_training = []
    labels_training = []
    labels_test = []

    training_data = data[data["case:concept:name"].isin(train_ids)]
    test_data = data[data["case:concept:name"].isin(test_ids)]
    # Create n-grams for training data

    for id_value, group in training_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)  # reset index for consistent slicing
        
        # Truncate group accordingly
        label = int(group['label'].dropna().iloc[0])

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str", "rule_1", "rule_2", "rule_3"])

        feature_names = group.columns.tolist()
        # Create n-grams of size 2, 4, 6, ...
        for n in range(1, len(group), 1):
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    # Create n-grams for test data
    for id_value, group in test_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)  # reset index for consistent slicing
        
        # Truncate group accordingly
        
        label = int(group['label'].dropna().iloc[0])
        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str", "rule_1", "rule_2", "rule_3"])
        # group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "lifecycle:transition", "concept:name_str"])
        
        feature_names = group.columns.tolist()
        # Create n-grams of size 2, 4, 6, ...
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    return ngrams_training, labels_training, ngrams_test, labels_test, feature_names

def preprocess_eventlog(data, dataset_size=None):

    vocab_sizes = {}
    cases = data[data["concept:name"] == "A_SUBMITTED-COMPLETE"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print(len(case_ids))
    print("Number of traces: ", len(labels))

    # train_ids, test_ids, _, _ = train_test_split(case_ids, labels, test_size=0.2, stratify=labels)
    # train_ids = admission_ids[:625]
    # test_ids = admission_ids[625:]
    # train_labels = labels[:625]
    # test_labels = labels[625:]
    train_ids, test_ids = create_test_set(data)

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in test set: ", len(test_ids))

    scaler_ar = MinMaxScaler()
    
    labels = data.groupby("case:concept:name")["label"].first().reset_index()
    print(data.columns)
    data = data.drop(columns=["case:REG_DATE"])
    # for col_name in data.columns.tolist():
    #     if col_name not in ["case:concept:name", "label", "concept:name", "org:group", "time:timestamp", "Diagnose", "lifecycle:transition", "LacticAcid", "CRP", "Leucocytes"]:
    #         data[col_name] = data[col_name].fillna(0).astype(int)

    data["concept:name_str"] = data["concept:name"]
    print("concept:name_str: ", data["concept:name_str"].unique())
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("W_Completeren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Completeren aanvraag-COMPLETE") + 1)
    print("W_Valideren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Valideren aanvraag-COMPLETE") + 1)
    print("O_SENT_BACK-COMPLETE: ", data["concept:name"].cat.categories.get_loc("O_SENT_BACK-COMPLETE") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = pd.Categorical(data["org:resource"])
    res_11169 = data["org:resource"].cat.categories.get_loc("11169") + 1
    print("11169 code: ", res_11169)
    res_10910 = data["org:resource"].cat.categories.get_loc("10910") + 1
    print("10910 code: ", res_10910)
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    # Numerical values
    data["case:AMOUNT_REQ"] = data["case:AMOUNT_REQ"].ffill()
    data["case:AMOUNT_REQ"] = scaler_ar.fit_transform(data[["case:AMOUNT_REQ"]])

    scalers = {
        "case:AMOUNT_REQ": scaler_ar
    }

    return create_ngrams(data, train_ids, test_ids), vocab_sizes, scalers