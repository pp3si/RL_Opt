import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import otrl

# Import states and rewards
states_df = pd.read_csv("pretrained_models/lunar_lander/ll_states_with_actions.csv", index_col=0)
rewards_df = pd.read_csv("pretrained_models/lunar_lander/ll_outcomes_with_actions.csv", index_col=0)

# Get best action for testing purposes
action_df = (rewards_df.eq(rewards_df.max(axis=1), axis=0)).astype(int)

# Train/test split
train_size = 10_000
states_train, states_test, rewards_train, rewards_test, action_train, action_test = train_test_split(
    states_df, rewards_df, action_df, test_size=(len(states_df.index) - train_size)
)

print(states_train)
print(rewards_train)
print(action_train)

# Define models
models_dict = {
    # "oct": otrl.OTRLClassifier(),
    # "opt": otrl.OTRLPolicy(),
    "opt_shelf": otrl.OTRLPolicyShelf()
}

# Train models
for name, model in models_dict.items():
    print(f"Training {name} . . .")
    model.fit(states_train, rewards_train, max_depths=[4, 5, 6], min_buckets=[0.005])
    model.save_model_json(f"iai_models/lunar_lander/json/{name}.json")
    model.save_tree_html(f"iai_models/lunar_lander/html/{name}.html")

    if name != "opt_shelf":
        print("AUC", roc_auc_score(
            action_test,
            model.lnr.predict_proba(states_test),
            multi_class="ovr",
            average="micro"
        ))
    print("Accuracy", accuracy_score(
        action_test.idxmax(axis=1),
        model.lnr.predict(states_test)
    ))

