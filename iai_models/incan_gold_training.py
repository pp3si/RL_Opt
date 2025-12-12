import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import otrl

# Import states

states_train_list = []
states_test_list = []
rewards_train_list = []
rewards_test_list = []
action_train_list = []
action_test_list = []

n_train_per_csv = 2_500
n_test_per_csv = 2_500

for n_player in tqdm(range(2, 9)):
    states_subdf = pd.read_csv(f"pretrained_models/incan_gold/incan_gold_states_{n_player}.csv", index_col=0)    
    rewards_subdf = pd.read_csv(f"pretrained_models/incan_gold/incan_gold_outcomes_{n_player}.csv", index_col=0)
    action_subdf = (rewards_subdf.eq(rewards_subdf.max(axis=1), axis=0)).astype(int)

    inds = np.random.choice(states_subdf.index, size=(n_train_per_csv + n_test_per_csv), replace=False)
    train_inds = inds[:n_train_per_csv]
    test_inds = inds[n_train_per_csv:]

    states_train_list.append(states_subdf.loc[train_inds])
    states_test_list.append(states_subdf.loc[test_inds])
    rewards_train_list.append(rewards_subdf.loc[train_inds])
    rewards_test_list.append(rewards_subdf.loc[test_inds])
    action_train_list.append(action_subdf.loc[train_inds])
    action_test_list.append(action_subdf.loc[test_inds])

states_train = pd.concat(states_train_list, axis=0, ignore_index=True)
states_test = pd.concat(states_test_list, axis=0, ignore_index=True)
rewards_train = pd.concat(rewards_train_list, axis=0, ignore_index=True)
rewards_test = pd.concat(rewards_test_list, axis=0, ignore_index=True)
action_train = pd.concat(action_train_list, axis=0, ignore_index=True)
action_test = pd.concat(action_test_list, axis=0, ignore_index=True)

print(states_train)
print(rewards_train)
print(action_train)

# Define models
models_dict = {
    "oct": otrl.OTRLClassifier(),
    "opt": otrl.OTRLPolicy(),
    "opt_shelf": otrl.OTRLPolicyShelf()
}

# Train models
for name, model in models_dict.items():
    print(f"Training {name} . . .")
    model.fit(states_train, rewards_train, max_depths=[4, 5, 6], min_buckets=[0.005])
    model.save_model_json(f"iai_models/incan_gold/json/{name}.json")
    model.save_tree_html(f"iai_models/incan_gold/html/{name}.html")

    if name != "opt_shelf":
        print("AUC", roc_auc_score(
            action_test["leave"],
            model.lnr.predict_proba(states_test)["leave"],
        ))
    print("Accuracy", accuracy_score(
        action_test.idxmax(axis=1),
        model.lnr.predict(states_test)
    ))

