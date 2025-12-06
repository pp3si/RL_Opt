import pandas as pd
import numpy as np
import json
import imageio

class OptimalTreeReinforcementLearning:

    def fit(self, observation, action_rewards,
            max_depths=None, min_buckets=None, hyperplane_config=None):
        pass

    def fit_oct(self, observation, target_actions, sample_weights=None,
                max_depths=None, min_buckets=None, hyperplane_config=None):

        from interpretableai import iai

        max_depths = max_depths or range(1, 8)
        min_buckets = min_buckets or [1]

        if hyperplane_config is None:
            grid = iai.GridSearch(
                iai.OptimalTreeClassifier(
                    weighted_minbucket=0
                ),
                max_depth=max_depths,
                minbucket=min_buckets
            )
        else:
            grid = iai.GridSearch(
            iai.OptimalTreeClassifier(
                weighted_minbucket=0,
                hyperplane_config=hyperplane_config
            ),
            max_depth=max_depths,
            minbucket=min_buckets
        )

        grid.fit(observation, target_actions, sample_weight=sample_weights)
        self.lnr = grid.get_learner()

    def save_model_json(self, filename="model.json"):
        self.lnr.write_json(filename)
        with open(filename, 'r') as file:
            data = json.load(file)
        data["rl_state_names"] = self.state_names
        data["rl_action_names"] = self.action_names
        with open(filename, 'w') as file:
            json.dump(data, file)

    def save_tree_html(self, filename):
        self.lnr.write_html(filename)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        # predict_input = pd.DataFrame({
        #     name: observation[ind]
        #     for ind, name in enumerate(self.state_names)
        #     }, index=[0])
        predict_input = pd.DataFrame(np.array(observation))
        if len(predict_input.columns) != len(self.state_names):
            predict_input = predict_input.T
            predict_input.columns = self.state_names
        else:
            predict_input.columns = self.state_names
        action = self.lnr.predict(predict_input)
        vectorized_action = [self.action_names_dict[act] for act in action]
        return vectorized_action
 
    def get_names(self, observation, action_rewards):
        self.state_names = list(observation.columns)
        self.action_names = list(action_rewards.columns)
        self.action_names_dict = {name: ind for ind, name in enumerate(self.action_names)}

    def render_episode(self, env, path="render_animation.gif", fps=20, loop=0):

        state, _ = env.reset()
        frames = [env.render()]
        while True:
            action = self.predict(state)
            state, _, done, _, _ = env.step(action[0])
            frames.append(env.render())
            if done:
                break
        imageio.mimwrite(path, frames, fps=fps, loop=loop)

    def evaluate(self, env, **kwargs):
        from stable_baselines3.common.evaluation import evaluate_policy
        self._original_predict = self.predict
        def _new_predict(*args, **kwargs):
            return self._original_predict(*args, **kwargs), None
        self.predict = _new_predict
        returns = evaluate_policy(self, env, **kwargs)
        self.predict = self._original_predict
        return returns

class OTRLClassifier(OptimalTreeReinforcementLearning):

    def fit(self, observation, action_rewards, max_depths=None, min_buckets=None, hyperplane_config=None):

        self.get_names(observation, action_rewards)

        target_actions = action_rewards.idxmax(axis=1)

        self.fit_oct(observation, target_actions, max_depths=max_depths, min_buckets=min_buckets, hyperplane_config=hyperplane_config)

class OTRLPolicy(OptimalTreeReinforcementLearning):

    def fit(self, observation, action_rewards, max_depths=None, min_buckets=None, hyperplane_config=None):

        self.get_names(observation, action_rewards)
        min_rewards = action_rewards.min(axis=1)

        observation_stack = pd.concat([observation.copy() for act in range(len(action_rewards.columns))], ignore_index=True)
        target_action_stack = pd.concat([pd.Series([act] * len(action_rewards.index)) for act in action_rewards.columns], ignore_index=True)
        sample_weights = pd.concat([(action_rewards[act] - min_rewards) for act in action_rewards.columns], ignore_index=True)
        
        keep_rows = (sample_weights > 0).tolist()
        observation_stack = observation_stack.loc[keep_rows, :]
        target_action_stack = target_action_stack.loc[keep_rows]
        sample_weights = sample_weights.loc[keep_rows]

        self.fit_oct(observation_stack, target_action_stack, sample_weights,
                     max_depths, min_buckets, hyperplane_config)
        
class OTRLPolicyShelf(OptimalTreeReinforcementLearning):

    def fit(self, observation, action_rewards, max_depths=None, min_buckets=None, hyperplane_config=None):

        from interpretableai import iai

        self.get_names(observation, action_rewards)

        max_depths = max_depths or range(1, 7)
        min_buckets = min_buckets or [1]

        if hyperplane_config is None:
            grid = iai.GridSearch(
                iai.OptimalTreePolicyMaximizer(),
                max_depth=max_depths,
                minbucket=min_buckets
            )
        else:
            grid = iai.GridSearch(
                iai.OptimalTreePolicyMaximizer(
                    hyperplane_config=hyperplane_config
                ),
                max_depth=max_depths,
                minbucket=min_buckets
            )

        grid.fit(observation, action_rewards)
        self.lnr = grid.get_learner()

class OTRLPretrained(OptimalTreeReinforcementLearning):

    def __init__(self, json_filename, load_full_iai=False):

        if load_full_iai:
            from interpretableai import iai
            self.lnr = iai.read_json(json_filename)
        else:
            from interpretableai import Predictor
            self.lnr = Predictor(json_filename)
        with open(json_filename, 'r') as file:
            data = json.load(file)
        self.state_names = data["rl_state_names"]
        self.action_names = data["rl_action_names"]
        self.action_names_dict = {name: ind for ind, name in enumerate(self.action_names)}

class EnumeratedPolicy(OptimalTreeReinforcementLearning):

    def __init__(self, observations, actions, state_names, action_names):

        self.policy = {
            obs: int(act)
            for obs, act in zip(observations, actions)
        }
        self.state_names = state_names
        self.action_names = action_names
        self.action_names_dict = {name: ind for ind, name in enumerate(self.action_names)}

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        observation = tuple([elm if not isinstance(elm, np.ndarray) else elm.item() for elm in observation])
        return [self.policy[observation]]

# import gymnasium as gym
# from stable_baselines3.common.monitor import Monitor

# # Import
# states_df = pd.read_csv("blackjack_states.csv")
# rewards_df = pd.read_csv("blackjack_outcomes.csv")

# # Train
# oct = OTRLPolicy()
# oct.fit(states_df, rewards_df, max_depths=[4, 5], min_buckets=[1])
# oct.lnr.write_html("tree.html")

# # Save
# oct.save_model_json("test.json")
# new_oct = OTRLPretrained("test.json")

# # Env
# env = Monitor(gym.make("Blackjack-v1", sab=True, render_mode="rgb_array"))

# # Eval
# print(new_oct.evaluate(env, n_eval_episodes=1_000))

# # Render
# new_oct.render_episode(env, fps=1)
