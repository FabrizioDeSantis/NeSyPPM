from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data):
    random.seed(42)
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())  # Get the unique group IDs
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * 0.3)

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 1)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 1)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 1)]

    filtered_values_nor = [v for v, l, r1, r2, r3 in zip(unique_groups, labels, labels_r1, labels_r2, labels_r3) if (r1 == 0 and r2 == 0 and r3 == 0 and l == 0)]
    filtered_values_r1_training = random.sample(filtered_values_r1, len(filtered_values_r1) // 2)
    filtered_values_r1_test = [x for x in filtered_values_r1 if x not in filtered_values_r1_training]
    filtered_values_r2_training = random.sample(filtered_values_r2, len(filtered_values_r2) // 2)
    filtered_values_r2_test = [x for x in filtered_values_r2 if x not in filtered_values_r2_training]
    filtered_values_r3_training = random.sample(filtered_values_r3, len(filtered_values_r3) // 2)
    filtered_values_r3_test = [x for x in filtered_values_r3 if x not in filtered_values_r3_training]

    print("Number of compliant rules in the training set: ", len(list(set(filtered_values_r1_training + filtered_values_r2_training + filtered_values_r3_training))))

    filtered_values = filtered_values_r1_test + filtered_values_r2_test + filtered_values_r3_test + filtered_values_nor
    filtered_values = list(set(filtered_values))  # Remove duplicates

    # take len_test_set from filtered_values
    if len(filtered_values) > len_test_set:
          # Set seed for reproducibility
        test_ids = random.sample(filtered_values, len_test_set)
    else:
        test_ids = filtered_values

    # Remove the selected test IDs from the original list
    training_ids = [x for x in unique_groups if x not in test_ids]
    training_ids = training_ids + filtered_values_r1_training + filtered_values_r2_training + filtered_values_r3_training
    training_ids = list(set(training_ids))  # Remove duplicates

    return training_ids, test_ids

def create_ngrams(data, train_ids, test_ids, window_size=10):

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp"])

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp"])
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

    data = data.dropna(subset=["case:concept:name"])
    vocab_sizes = {}
    cases = data[data["concept:name"] == "Create Fine"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print(len(case_ids))
    print("Number of traces: ", len(labels))

    scalers = {}
    scaler_amount = MinMaxScaler()
    scaler_paymentamount = MinMaxScaler()
    scaler_totalpaymentamount = MinMaxScaler()
    scaler_expense = MinMaxScaler()

    print(data.columns)

    train_ids, test_ids = create_test_set(data)
    #train_ids, test_ids, _, _ = train_test_split(case_ids, labels, test_size=0.2, stratify=labels)

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in test set: ", len(test_ids))

    data = data.drop(columns=["lifecycle:transition", "matricola", 'totalPaymentAmount'])
    data = data[['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource', 'dismissal', 'vehicleClass',
       'article', 'points', 'expense', 'notificationType',
       'lastSent', 'amount', 'paymentAmount', 'label']]

    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("Add penalty: ", data["concept:name"].cat.categories.get_loc("Add penalty") + 1)
    print("Create fine: ", data["concept:name"].cat.categories.get_loc("Create Fine") + 1)
    print("Send fine: ", data["concept:name"].cat.categories.get_loc("Send Fine") + 1)
    print("Payment: ", data["concept:name"].cat.categories.get_loc("Payment") + 1)
    print("Insert Fine Notification: ", data["concept:name"].cat.categories.get_loc("Insert Fine Notification") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = data["org:resource"].fillna(0)
    data["org:resource"] = pd.Categorical(data["org:resource"])
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    data["dismissal"] = data["dismissal"].ffill()
    data["dismissal"] = pd.Categorical(data["dismissal"])
    data["dismissal"] = data["dismissal"].cat.codes + 1
    vocab_sizes["dismissal"] = data["dismissal"].max()

    data["amount"] = data["amount"].ffill()
    data["amount"] = data["amount"].fillna(0)
    data["amount"] = scaler_amount.fit_transform(data[["amount"]])
    scalers["amount"] = scaler_amount

    data["lastSent"] = data["lastSent"].fillna(0)
    data["lastSent"] = pd.Categorical(data["lastSent"])
    data["lastSent"] = data["lastSent"].cat.codes + 1
    vocab_sizes["lastSent"] = data["lastSent"].max()

    data["paymentAmount"] = data["paymentAmount"].fillna(0)
    scaler_paymentamount = scaler_paymentamount.fit_transform(data[["paymentAmount"]])
    scalers["paymentAmount"] = scaler_paymentamount

    # data["tototalPaymentAmount"] = data["totalPaymentAmount"].fillna(0)
    # scaler_totalpaymentamount = scaler_totalpaymentamount.fit_transform(data[["totalPaymentAmount"]])

    data["points"] = data["points"].fillna(-1)
    data["points"] = pd.Categorical(data["points"])
    data["points"] = data["points"].cat.codes + 1
    vocab_sizes["points"] = data["points"].max()

    data["vehicleClass"] = data["vehicleClass"].fillna(0)
    data["vehicleClass"] = pd.Categorical(data["vehicleClass"])
    data["vehicleClass"] = data["vehicleClass"].cat.codes + 1
    vocab_sizes["vehicleClass"] = data["vehicleClass"].max()

    data["notificationType"] = data["notificationType"].fillna(0)
    data["notificationType"] = pd.Categorical(data["notificationType"])
    data["notificationType"] = data["notificationType"].cat.codes + 1
    vocab_sizes["notificationType"] = data["notificationType"].max()

    data["expense"] = data["expense"].fillna(0)
    data["expense"] = scaler_expense.fit_transform(data[["expense"]])
    scalers["expense"] = scaler_expense

    data["article"] = data["article"].ffill()
    data["article"] = pd.Categorical(data["article"])
    data["article"] = data["article"].cat.codes + 1
    vocab_sizes["article"] = data["article"].max()

    return create_ngrams(data, train_ids, test_ids), vocab_sizes, scalers