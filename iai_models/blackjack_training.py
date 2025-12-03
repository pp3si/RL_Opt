import pandas as pd
import otrl

# Import states and rewards
states_df = pd.read_csv("pretrained_models/blackjack/blackjack_states.csv")
rewards_df = pd.read_csv("pretrained_models/blackjack/blackjack_outcomes.csv")

# Define models
models_dict = {
    "oct": otrl.OTRLClassifier(),
    "opt": otrl.OTRLPolicy(),
    "opt_shelf": otrl.OTRLPolicyShelf()
}

# Train models
for name, model in models_dict.items():
    print(f"Training {name} . . .")
    model.fit(states_df, rewards_df, max_depths=range(5), min_buckets=[1])
    model.save_model_json(f"iai_models/trained/blackjack/{name}.json")
