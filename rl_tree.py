from interpretableai import iai
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import imageio

class OptimalTreeReinforcementLearning:

    def fit(self, observation, action_rewards,
            max_depths=None, min_buckets=None):
        pass

    def fit_oct(self, observation, target_actions, sample_weights=None,
                max_depths=None, min_buckets=None):

        max_depths = max_depths or range(1, 7)
        min_buckets = min_buckets or [1]

        grid = iai.GridSearch(
            iai.OptimalTreeClassifier(
                weighted_minbucket=0
            ),
            max_depth=max_depths,
            minbucket=min_buckets
        )

        grid.fit(observation, target_actions, sample_weight=sample_weights)
        self.lnr = grid.get_learner()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        predict_input = pd.DataFrame({
            name: observation[ind]
            for ind, name in enumerate(self.state_names)
            }, index=[0])
        action = self.lnr.predict(predict_input)
        vectorized_action = [self.action_names_dict[act] for act in action]        
        return vectorized_action, None
    
    def get_names(self, observation, action_rewards):
        self.state_names = observation.columns
        self.action_names = action_rewards.columns
        self.action_names_dict = {name: ind for ind, name in enumerate(self.action_names)}

    def render_episode(self, env, path="render_animation.gif", fps=20, loop=0):

        state, _ = env.reset()
        frames = [env.render()]
        while True:
            action, _ = self.predict(state)
            state, _, done, _, _ = env.step(action[0])
            frames.append(env.render())
            if done:
                break
        imageio.mimwrite(path, frames, fps=fps, loop=loop)

class OTRLClassifier(OptimalTreeReinforcementLearning):

    def fit(self, observation, action_rewards, max_depths=None, min_buckets=None):

        self.get_names(observation, action_rewards)

        target_actions = action_rewards.idxmax(axis=1)

        self.fit_oct(observation, target_actions, max_depths=max_depths, min_buckets=min_buckets)

class OTRLPolicy(OptimalTreeReinforcementLearning):

    def fit(self, observation, action_rewards, max_depths=None, min_buckets=None):

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
                     max_depths, min_buckets)

states_df = pd.read_csv("blackjack_states.csv")
rewards_df = pd.read_csv("blackjack_outcomes.csv")

oct = OTRLPolicy()
oct.fit(states_df, rewards_df, max_depths=[4, 5], min_buckets=[1])
oct.lnr.write_html("tree.html")

env = Monitor(gym.make("Blackjack-v1", sab=True, render_mode="rgb_array"), "eval_logs/")

mean_reward, std_reward = evaluate_policy(oct, env, n_eval_episodes=1_000)

print(mean_reward, std_reward)

oct.render_episode(env, fps=1)
