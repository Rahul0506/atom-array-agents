from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gym_atom_array.env import ArrayEnv, Config
from extractors import GridExtractor


TIMESTEPS = 5_000_000

size = 5
ROWS, COLS = size, size
small_grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
config = Config(DefaultPenalty=0.01, TargetRelease=5, TargetPickUp=10)
env_arg_dict = dict(n_rows=size, n_cols=size, targets=small_grid, config=config)

def make_env():
    env = ArrayEnv(n_rows=size, n_cols=size, targets=small_grid, config=config)
    env = TimeLimit(env, 200)  # new time limit
    return env

# Parallel environments
env = make_vec_env(make_env, n_envs=4)

policy_kwargs = {
    'features_extractor_class': GridExtractor,
    'net_arch': [64, dict(pi=[32, 16], vf=[32, 16])],
    }
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=TIMESTEPS)
model.save("ppo_array1")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_array1")

env = make_env()
obs = env.reset()
# while True:
for _ in range(30):
    action, _states = model.predict(obs)
    print("Action: " + "UDLRAB"[action])
    obs, rewards, dones, info = env.step(action)
    env.render()
