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

for n_player in range(2, 9):
    states_subdf = pd.read_csv(f"pretrained_models/incan_gold/incan_gold_states_{n_player}.csv", index_col=0)
    
    states_subdf["risky_cards"] = (
        ((states_subdf["snakes_revealed"] >= 1) * (states_subdf["snakes_remaining"] - states_subdf["snakes_revealed"]))
        + ((states_subdf["mummies_revealed"] >= 1) * (states_subdf["mummies_remaining"] - states_subdf["mummies_revealed"]))
        + ((states_subdf["spiders_revealed"] >= 1) * (states_subdf["spiders_remaining"] - states_subdf["spiders_revealed"]))
        + ((states_subdf["rocks_revealed"] >= 1) * (states_subdf["rocks_remaining"] - states_subdf["rocks_revealed"]))
        + ((states_subdf["fires_revealed"] >= 1) * (states_subdf["fires_remaining"] - states_subdf["fires_revealed"]))
    )
    # states_df["my_curr"] = states_df[[f"player_{ind+1}_curr" for ind in range(8)]].iloc[:, states_df["player_id"]]
    states_subdf["my_curr"] = 0
    for ind in tqdm(states_subdf.index):
        player_ind = int(states_subdf.loc[ind, "player_id"] + 1)
        states_subdf.loc[ind, "my_curr"] = states_subdf.loc[ind, f"player_{player_ind}_curr"]
    states_subdf["players_in"] = (
        states_subdf[[f"player_{ind+1}_in" for ind in range(8)]]
    ).sum(axis=1)    
    
    rewards_subdf = pd.read_csv(f"pretrained_models/incan_gold/incan_gold_outcomes_{n_player}.csv", index_col=0)
    action_subdf = (rewards_subdf.eq(rewards_subdf.max(axis=1), axis=0)).astype(int)
    states_train_subdf, states_test_subdf, rewards_train_subdf, rewards_test_subdf, action_train_subdf, action_test_subdf = train_test_split(
        states_subdf, rewards_subdf, action_subdf, test_size=(len(states_subdf.index) - n_train_per_csv)
    )

    states_train_list.append(states_train_subdf)
    states_test_list.append(states_test_subdf)
    rewards_train_list.append(rewards_train_subdf)
    rewards_test_list.append(rewards_test_subdf)
    action_train_list.append(action_train_subdf)
    action_test_list.append(action_test_subdf)



states_train = pd.concat(states_train_list, axis=0, ignore_index=True)
states_test = pd.concat(states_test_list, axis=0, ignore_index=True)
rewards_train = pd.concat(rewards_train_list, axis=0, ignore_index=True)
rewards_test = pd.concat(rewards_test_list, axis=0, ignore_index=True)
action_train = pd.concat(action_train_list, axis=0, ignore_index=True)
action_test = pd.concat(action_test_list, axis=0, ignore_index=True)


# states_list = [
#     pd.read_csv(f"pretrained_models/incan_gold/incan_gold_states_{n_player}.csv", index_col=0)
#     for n_player in range(2, 9)
# ]
# states_df = pd.concat(states_list, axis=0, ignore_index=True)
# states_df["risky_cards"] = (
#     np.maximum((states_df["snakes_revealed"] >= 1) * (states_df["snakes_remaining"] - states_df["snakes_revealed"]), 0)
#     + np.maximum((states_df["mummies_revealed"] >= 1) * (states_df["mummies_remaining"] - states_df["mummies_revealed"]), 0)
#     + np.maximum((states_df["spiders_revealed"] >= 1) * (states_df["spiders_remaining"] - states_df["spiders_revealed"]), 0)
#     + np.maximum((states_df["rocks_revealed"] >= 1) * (states_df["rocks_remaining"] - states_df["rocks_revealed"]), 0)
#     + np.maximum((states_df["fires_revealed"] >= 1) * (states_df["fires_remaining"] - states_df["fires_revealed"]), 0)
# )
# # states_df["my_curr"] = states_df[[f"player_{ind+1}_curr" for ind in range(8)]].iloc[:, states_df["player_id"]]
# states_df["my_curr"] = 0
# for ind in tqdm(states_df.index):
#     player_ind = int(states_df.loc[ind, "player_id"] + 1)
#     states_df.loc[ind, "my_curr"] = states_df.loc[ind, f"player_{player_ind}_curr"]
# states_df["players_in"] = (
#     states_df[[f"player_{ind+1}_in" for ind in range(8)]]
# ).sum(axis=1)
# print(states_df)

# # Import rewards
# rewards_list = [
#     pd.read_csv(f"pretrained_models/incan_gold/incan_gold_outcomes_{n_player}.csv", index_col=0)
#     for n_player in range(2, 9)
# ]
# rewards_df = pd.concat(rewards_list, axis=0, ignore_index=True)
# # print(rewards_df)

# # Get best action for testing purposes
# action_df = (rewards_df.eq(rewards_df.max(axis=1), axis=0)).astype(int)

# # Train/test split
# train_size = 20_000
# states_train, states_test, rewards_train, rewards_test, action_train, action_test = train_test_split(
#     states_df, rewards_df, action_df, test_size=(len(states_df.index) - train_size)
# )

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
    model.save_model_json(f"iai_models/incan_gold_modified/json/{name}.json")
    model.save_tree_html(f"iai_models/incan_gold_modified/html/{name}.html")

    if name != "opt_shelf":
        print("AUC", roc_auc_score(
            action_test["leave"],
            model.lnr.predict_proba(states_test)["leave"],
        ))
    print("Accuracy", accuracy_score(
        action_test.idxmax(axis=1),
        model.lnr.predict(states_test)
    ))

