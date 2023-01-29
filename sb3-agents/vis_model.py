from gym.wrappers import TimeLimit
from gym_atom_array.env import ArrayEnv, Config

from stable_baselines3 import PPO

size = 5
ROWS, COLS = size, size
small_grid = [(1, 1), (1, 2), (1, 3)]
config = Config(DefaultPenalty=0.01, TargetRelease=0.1)
env_arg_dict = dict(n_rows=size, n_cols=size, targets=small_grid, config=config)

def make_env():
    env = ArrayEnv(n_rows=size, n_cols=size, targets=small_grid, config=config)
    env = TimeLimit(env, 200)  # new time limit
    return env


model = PPO.load("ppo_array")

env = make_env()
obs = env.reset()
# while True:
for _ in range(30):
    action, _states = model.predict(obs)
    print("Action: " + "UDLRAB"[action])
    obs, rewards, dones, info = env.step(action)
    env.render()
