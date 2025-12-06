import gymnasium as gym
import torch
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from lunar_lander_env import ModifiedLunarLander

'''
The pretrained Lunar Lander DQN model from StableBaselines3 Zoo is too good.
The agent is heavily rewarded for each step where the lander's legs are touching the ground.
The SB3 model exploits this by
    (1) landing the lander
    (2) firing the side engines to "jitter" the lander's position just enough to prevent the environment from terminating
    (3) farming reward for having numerous steps where the lander's legs are touching the ground

This script retrains the model under a new environoment that penalizes engine firing when touching the ground.

Co-authored with code from lunar_lander_rl.ipynb and ChatGPT
'''

# class PenalizeFiringOnGround(gym.Wrapper):
#     """
#     Severely penalizes the lander if it fires any engines while on the ground
#     Terminates the episode immediately when the LunarLander environment
#     detects a safe landing (+100 reward) or a crash (-100 reward).
#     Give some tolerance to reward threshold (+-50 reward) in case the lunar lander gets points elsewhere. 
#     """

#     def step(self, action):

#         state, reward, terminated, truncated, info = self.env.step(action)
#         firing_on_ground = (action > 0) * (sum(state[-2:]) == 2)
#         return state, reward - 5 * firing_on_ground, terminated, truncated, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

class StateRecorderCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_states = []
        self.all_episode_states = []

    def _on_step(self) -> bool:
        # obs from VecEnv is shape (n_envs, obs_dim)
        obs = self.locals['new_obs'][0].copy()
        self.episode_states.append(obs)

        dones = self.locals['dones']  # array of bools for each env
        if dones[0]:
            # episode finished
            self.all_episode_states.append(self.episode_states)
            self.episode_states = []

        return True

# Import model
model_path = "rl-baselines3-zoo/rl-trained-agents/dqn/LunarLander-v2_1/LunarLander-v2.zip"
model = DQN.load(model_path, exploration_initial_eps=0.0, exploration_final_eps=0.0)

# # Force fully greedy actions
# model.exploration_rate = 0.0

# Create environment with early-termination wrapper
model.env = DummyVecEnv([lambda : ModifiedLunarLander()])

# # Wrap environment as a VecEnv
# vec_env = make_vec_env(lambda: env, n_envs=1)
# model.set_env(vec_env)

# Create the callback
state_recorder = StateRecorderCallback()

# Train
model.learn(total_timesteps=(10 ** 5), callback=state_recorder)

# Retrieve all states
all_episode_states = state_recorder.all_episode_states

# Save updated model
model.save("pretrained_models/lunar_lander/lunar_lander_retrained.zip")

all_states_rows = []  # Will store rows for pandas
all_qvalues_rows = []

for ep_idx, episode in enumerate(all_episode_states):
    for step_idx, state in enumerate(episode):
        # Convert state to tensor for PyTorch
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape (1, obs_dim)
        
        # Get Q-values from the DQN model
        with torch.no_grad():
            q_values = model.q_net(state_tensor)  # shape (1, n_actions)
        q_values_np = q_values.numpy().flatten()
        
        # Record state and qvalues with metadata
        state_row = {**{f'state_{i}': s for i, s in enumerate(state)}}
        qvalue_row = {**{f'qvalue_{i}': q for i, q in enumerate(q_values_np)}}
        
        all_states_rows.append(state_row)
        all_qvalues_rows.append(qvalue_row)

# Convert to pandas DataFrames
states_df = pd.DataFrame(all_states_rows)
states_df.columns = [
    "x",
    "y",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_contact",
    "right_contact"
]
qvalues_df = pd.DataFrame(all_qvalues_rows)
qvalues_df.columns = [
    "Nothing",
    "Left",
    "Main",
    "Right"
]
print(states_df)
print(qvalues_df)

# Save to CSV if needed
states_df.to_csv("pretrained_models/lunar_lander/lunar_lander_states_retrained.csv", index=False)
qvalues_df.to_csv("pretrained_models/lunar_lander/lunar_lander_outcomes_retrained.csv", index=False)


# # Define number of episodes
# EPISODES = 1

# all_episode_states = []  # stores list of lists: states per episode

# # Train
# for episode in range(EPISODES):
#     state, _ = env.reset()
#     episode_states = []
#     done = False

#     while not done:
#         episode_states.append(state)

#         # Convert state to tensor (batch dimension 1)
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

#         # Compute Q-values with no gradient
#         with torch.no_grad():
#             q_values = model.q_net(state_tensor).cpu().numpy()  # shape: (1, n_actions)

#         # print(q_values)

#         # Take argmax across actions, then convert to Python int
#         action = np.argmax(q_values[0])

#         # print(action)

#         # Step environment
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         # Replay buffer
#         model.replay_buffer.add(
#             state,
#             next_state,
#             action,
#             reward,
#             done,
#             [{}]
#         )
#         # Train if replay buffer has enough samples
#         if model.replay_buffer.size() > model.batch_size:
#             # model.train(batch_size=model.batch_size, gradient_steps=1)
#             model.learn(total_timesteps=0, reset_num_timesteps=False)

#         state = next_state

#     all_episode_states.append(episode_states)
#     print(f"Episode {episode+1} finished, states recorded: {len(episode_states)}")

# env.close()

# # Save model
# model.save("lunar_lander_retrained", include_pkl=False)

# # Compute q-values
# all_qvalues = []

# for episode_states in all_episode_states:
#     episode_q = []
#     for state in episode_states:

#         # convert to torch tensor
#         state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

#         # compute Q-values using model.q_net
#         with torch.no_grad():
#             q_values = model.q_net(state_t).cpu().numpy().flatten()

#         episode_q.append(q_values)

#     all_qvalues.append(episode_q)

# # Save q-values
# states = pd.DataFrame(
#     all_episode_states,
#     columns=[
#         "x,",
#              "y",
#              "x'",
#              "y'",
#              "angle",
#              "angular_velocity",
#              "left_contact",
#              "right_contact"
#     ]
# )
# states.to_csv("lunar_lander_states_retrained.csv")
# outcomes = pd.DataFrame(
#     all_qvalues,
#     columns=[
#         "Nothing",
#         "Left",
#         "Main",
#         "Right"
#     ]
# )
# outcomes.to_csv("lunar_lander_outcomes_retrained.csv")
