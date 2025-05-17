import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_group_subset(df, group_col, label_col, percentage):

    if not 0 < percentage < 1:
        print("Warning: Percentage must be between 0 and 1. Returning empty DataFrame.")
        return pd.DataFrame()
    
    if df.empty:
        print("Warning: Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    if group_col not in df.columns or label_col not in df.columns:
        raise ValueError("Specified group_col or label_col not found in DataFrame.")


    # 1. Group by the group column and get unique groups
    grouped = df.groupby(group_col)
    unique_groups = list(grouped.groups.keys())  # Get the unique group IDs

    # Create a Series to hold the label for each group (taking the first label within each group is sufficient)
    group_labels = df.groupby(group_col)[label_col].first()


    # 2. Calculate the label distribution in the original data
    label_counts = group_labels.value_counts(normalize=True)

    # 3. Calculate the number of groups to select for each label
    num_groups_to_select = int(len(unique_groups) * percentage)
    groups_per_label = (label_counts * num_groups_to_select).round().astype(int)

     # Ensure at least one group is selected.  Important if num_groups_to_select is small and some labels are rare.
    for label in groups_per_label.index:
        if groups_per_label[label] == 0 and label in group_labels.values:
            groups_per_label[label] = 1
            # Adjust other counts to compensate, prioritizing the most frequent labels.
            max_label = groups_per_label.idxmax()
            if groups_per_label[max_label]>1: #only adjust if other category as more than 1
                groups_per_label[max_label] -= 1
    
    # Check if we are still selecting the same number of groups
    if groups_per_label.sum() < num_groups_to_select:
        max_label = groups_per_label.idxmax()
        groups_per_label[max_label] += num_groups_to_select - groups_per_label.sum()
    elif groups_per_label.sum() > num_groups_to_select:
        max_label = groups_per_label.idxmax()
        groups_per_label[max_label] -= groups_per_label.sum() - num_groups_to_select
      

    # 4. Sample groups for each label
    selected_groups = []
    for label, count in groups_per_label.items():
        groups_with_label = group_labels[group_labels == label].index.tolist()

        # Handle cases where there are fewer groups with the label than requested
        num_to_sample = min(count, len(groups_with_label))
        if num_to_sample > 0: #avoid error with random.choice when sampling 0 elements
             selected_groups.extend(np.random.choice(groups_with_label, size=num_to_sample, replace=False))

    # 5. Filter the original DataFrame based on selected groups
    subset_df = df[df[group_col].isin(selected_groups)]

    return subset_df

def generate_ngrams(group, one_hot_dict, one_hots, max_length=30):
    """Generate n-grams of up to max_length for a group of concept:name values."""
    ngrams = []
    for i in range(len(group)):
        l = []
        # Create n-gram sequences of increasing length up to max_length
        for act in group[:i + 1].tolist()[-max_length:]:
            l.append(one_hots[one_hot_dict[act]].tolist())
        ngrams.append(l)
    return ngrams

def generate_ngrams_integers(group, max_length=25):
    """Generate n-grams of up to max_length for a group of concept:name values."""

    ngrams = [None] * len(group["concept:name"])
    ngrams_lg = [None] * len(group["concept:name"])
    ngrams_lf = [None] * len(group["concept:name"])
    n_grams_am = [None] * len(group["concept:name"])
    n_grams_type = [None] * len(group["concept:name"])
    n_grams_offer = [None] * len(group["concept:name"])
    n_grams_res = [None] * len(group["concept:name"])
    n_grams_mc = [None] * len(group["concept:name"])
    n_grams_cs = [None] * len(group["concept:name"])
    n_grams_elapday = [None] * len(group["concept:name"])
    n_grams_hour = [None] * len(group["concept:name"])

    concept_name = np.array(group["concept:name"])
    loan_goal = np.array(group["case:LoanGoal"])
    lifecycle = np.array(group["lifecycle:transition"])
    app_type = np.array(group["case:ApplicationType"])
    resource = np.array(group["org:resource"])
    amount = np.array(group["case:RequestedAmount"])
    offer = np.array(group["OfferedAmount"])
    monthcost = np.array(group["MonthlyCost"])
    creditscore = np.array(group["CreditScore"])
    elapday = np.array(group["elapsed_days"])
    hour = np.array(group["hour_of_day"])

    for i in range(len(concept_name)):
        
        start_index = max(0, i + 1 - max_length)
        ngrams[i] = concept_name[start_index:i + 1].tolist()
        ngrams_lg[i] = loan_goal[start_index:i + 1].tolist()
        ngrams_lf[i] = lifecycle[start_index:i + 1].tolist()
        n_grams_type[i] = app_type[start_index:i + 1].tolist()
        n_grams_am[i] = amount[start_index:i + 1].tolist()
        n_grams_offer[i] = offer[start_index:i + 1].tolist()
        n_grams_res[i] = resource[start_index:i + 1].tolist()
        n_grams_mc[i] = monthcost[start_index:i + 1].tolist()
        n_grams_cs[i] = creditscore[start_index:i + 1].tolist()
        n_grams_elapday[i] = elapday[start_index:i + 1].tolist()
        n_grams_hour[i] = hour[start_index:i + 1].tolist()

    # return ngrams, ngrams_lf, n_grams_am, n_grams_offer, n_grams_type, n_grams_res, n_grams_mc, n_grams_elapday, n_grams_hour, n_grams_cs
    return ngrams, ngrams_lg, ngrams_lf, n_grams_am, n_grams_offer, n_grams_type, n_grams_res, n_grams_mc, n_grams_cs

def create_prefixes_integer(data, dataset_size):
    all_seqlab = []
    labels = []
    count = 0
    if dataset_size is not None:
        data = stratified_group_subset(data, "case:concept:name", "label", dataset_size/100)
    groups = data.groupby('case:concept:name')
    #subset = stratified_group_subset(data, "case:concept:name", "label", 0.2)
    print("Number of cases:", len(groups))
    # print("100%: ", len(subset.groupby('case:concept:name')))
    for name, group_data in data.groupby('case:concept:name'):
        label, label_cs = group_data["label"].iat[0], group_data["label_ra"].iat[0]
        # prefixes = [(ngram, ngram_lf, gram_apptype, ngram_resource, ngram_amount, 
        #                    ngram_offer, ngram_mc, ngram_ed, ngram_hd, ngram_cs, label_cs, name) for 
        #                    ngram, ngram_lf, ngram_amount, ngram_offer, gram_apptype, 
        #                    ngram_resource, ngram_mc, ngram_ed, ngram_hd, ngram_cs in zip(*generate_ngrams_integers(group_data))]
        prefixes = [(ngram, ngram_lg, ngram_lf, gram_apptype, ngram_resource, ngram_amount, 
                           ngram_offer, ngram_mc, ngram_cs, label_cs, name) for 
                           ngram, ngram_lg, ngram_lf, ngram_amount, ngram_offer, gram_apptype, 
                           ngram_resource, ngram_mc, ngram_cs in zip(*generate_ngrams_integers(group_data))]
        all_seqlab.extend(prefixes)
        labels.extend([label] * len(prefixes))

        # if count == 100:
        #     break
        # count += 1
    return all_seqlab, labels

def preprocess_data_rob(data, dataset_size=None):
    return create_prefixes_integer(data, dataset_size), (21, 7, 2, 148, 14)

def preprocess_data_integer(data, dataset_size=None):

    scaler = MinMaxScaler()
    scaler_requested_amount = MinMaxScaler()
    scaler_creditscore = MinMaxScaler()

    mask = data["concept:name"].str.startswith("O_")
    # Forward fill within each group, but only for masked rows
    data["OfferedAmount"] = data.groupby("case:concept:name")["OfferedAmount"].ffill().where(mask)
    data["OfferedAmount"] = data["OfferedAmount"].fillna(0)

    data['concept:name'] = pd.Categorical(data['concept:name'])
    o_create_offert_code = data['concept:name'].cat.categories.get_loc("O_Create Offer") + 1
    data['concept:name'] = data['concept:name'].cat.codes + 1
    print("O_Create Offert code:", o_create_offert_code)

    data['lifecycle:transition'] = pd.Categorical(data['lifecycle:transition'])
    data['lifecycle:transition'] = data['lifecycle:transition'].cat.codes + 1

    data['case:LoanGoal'] = pd.Categorical(data['case:LoanGoal'])
    home_impr = data['case:LoanGoal'].cat.categories.get_loc("Existing loan takeover") + 1
    data['case:LoanGoal'] = data['case:LoanGoal'].cat.codes + 1
    print("Home improvement:", home_impr)
    data['case:ApplicationType'] = pd.Categorical(data['case:ApplicationType'])
    data['case:ApplicationType'] = data['case:ApplicationType'].cat.codes + 1

    data['org:resource'] = pd.Categorical(data['org:resource'])
    data['org:resource'] = data['org:resource'].cat.codes + 1
    
    data['case:RequestedAmount'] = scaler_requested_amount.fit_transform(data[['case:RequestedAmount']])
    #data['OfferedAmount'] = data['OfferedAmount'].fillna(0)
    data['OfferedAmount'] = scaler_creditscore.fit_transform(data[['OfferedAmount']])

    data['CreditScore'] = data['CreditScore'].fillna(0)
    data['CreditScore'] = scaler.fit_transform(data[['CreditScore']])

    data["MonthlyCost"] = data.groupby("case:concept:name")["MonthlyCost"].ffill().where(mask)
    data["MonthlyCost"] = data["MonthlyCost"].fillna(0)
    data['MonthlyCost'] = scaler.fit_transform(data[['MonthlyCost']])

    # data.to_csv("data_processed/dataset_processed.csv", index=False)

    # with open("parameters.txt", "w") as f:
    #     f.write(f"{data['concept:name'].max()}\n")
    #     f.write(f"{data['lifecycle:transition'].max()}\n")
    #     f.write(f"{data['case:ApplicationType'].max()}\n")
    #     f.write(f"{data['org:resource'].max()}\n")
    #     f.write(f"{data['case:LoanGoal'].max()}\n")
    # exit()
    return create_prefixes_integer(data, dataset_size), scaler_requested_amount, scaler_requested_amount, (data['concept:name'].max(), data["lifecycle:transition"].max(), data["case:ApplicationType"].max(), data["org:resource"].max(), data["case:LoanGoal"].max())