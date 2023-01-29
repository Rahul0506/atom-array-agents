from stable_baselines3.common.env_checker import check_env

from gym_atom_array.env import ArrayEnv, Config

size = 5
ROWS, COLS = size, size
small_grid = [(1, 1), (1, 2), (1, 3)]
config = Config(Render=True)

env = ArrayEnv(n_rows=ROWS, n_cols=COLS, targets=small_grid, config=config)

check_env(env)
