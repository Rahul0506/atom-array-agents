import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gym_atom_array.env import ArrayEnv
from clean_agents.ppo import make_env
from clean_agents.networks import SeparateNets as Agent

from random import randint
from sys import argv
from argparse import Namespace

print(argv)
assert len(argv) == 3
wandb_name = argv[1]
model_version = argv[2]

model_path = f"wandb/{wandb_name}/files/agent-{model_version}.pt"

args = Namespace(
    Render=False,
    ArraySize=6,
    TargetSize=4,
    DefaultPenalty=-0.1,
    TargetPickUp=-5,
    TargetRelease=10,
    TimeLimit=200,
)
envs = gym.vector.SyncVectorEnv(
    [
        make_env(randint(0, 100), args),
    ]
)


device = torch.device("cpu")
agent = Agent(envs, device)
state_dict = torch.load(model_path)
agent.load_state_dict(state_dict)
agent.eval()


def obs_to_plots(obs, atoms_plt, mt_plt):
    atom_grid, tar_grid, mt_grid = obs[0]

    dots = [[], []]
    mt_pos = (0, 0)
    for i in range(n):
        for j in range(n):
            if mt_grid[i, j] != 0:
                mt_pos = (i, j, mt_grid[i, j])
            if atom_grid[i, j] == 1:
                dots[0].append(i)
                dots[1].append(j)
    atoms_plt.set_data(dots)
    mt_plt.set_data(mt_pos[0], mt_pos[1])
    mt_plt.set_marker("x" if mt_pos[2] == 1 else "o")

    return atoms_plt, mt_plt


fig, ax = plt.subplots(figsize=(8, 8))

n = args.ArraySize
ax.set_xlim(-1, n)
ax.set_ylim(-1, n)

obs = envs.reset()
atom_grid, tar_grid, mt_grid = obs[0]
dots = [[], []]
for i in range(n):
    for j in range(n):
        if tar_grid[i, j] == 1:
            dots[0].append(i)
            dots[1].append(j)

frame_title = ax.set_title("Frame: 0")
(targets,) = ax.plot(
    dots[0], dots[1], "gs", markersize=12, markerfacecolor=(1, 1, 0, 0.5)
)
(atoms,) = ax.plot([], [], "bo")
(mt,) = ax.plot(0, 0, "ro", markersize=10)

obs_to_plots(obs, atoms, mt)

next_obs = torch.Tensor(obs).to(device)
done = False


def animate(frame_num):
    global done, next_obs, atoms, mt, frame_title

    frame_title.set_text(f"Frame: {frame_num}")
    with torch.no_grad():
        action, logprob, _, _ = agent.get_action_and_value(next_obs)
        action = action.cpu().numpy()
        print(action)
        next_obs, reward, done, info = envs.step(action)

    if done:
        return ()

    atoms, mt = obs_to_plots(next_obs, atoms, mt)
    next_obs = torch.Tensor(next_obs).to(device)
    return (atoms, mt, frame_title)


def done_gen():
    global done, args
    i = 0
    while not done and i < args.TimeLimit:
        i += 1
        yield i


anim = FuncAnimation(fig, animate, frames=done_gen, interval=100, blit=True)

plt.show()
