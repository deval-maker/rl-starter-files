import gym
import gym_minigrid
import gym_multigrid

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
