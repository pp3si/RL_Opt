import gymnasium as gym
'''
The pretrained Lunar Lander DQN model from StableBaselines3 Zoo is too good.
The agent is heavily rewarded for each step where the lander's legs are touching the ground.
The SB3 model exploits this by
    (1) landing the lander
    (2) firing the side engines to "jitter" the lander's position just enough to prevent the environment from terminating
    (3) farming reward for having numerous steps where the lander's legs are touching the ground

This script redefines the environment so that the agent receives -5 reward if it fires engines while on the ground.

Co-authored with code from lunar_lander_rl.ipynb and ChatGPT
'''

class PenalizeFiringOnGround(gym.Wrapper):
    """
    Severely penalizes the lander if it fires any engines while on the ground
    """
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        firing_on_ground = (action > 0) * (sum(state[-2:]) == 2)
        return state, reward - 5 * firing_on_ground, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ModifiedLunarLander(gym.Env):
    def __init__(self):
        # Create the original environment
        self.env = gym.make("LunarLander-v3", render_mode="rgb_array")
        
        # Wrap it with the reward modifier
        self.env = PenalizeFiringOnGround(self.env)
        
        # Expose the gym.Env interface
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()
