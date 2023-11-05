import random
import numpy as np
from .agent import Agent

def generate_stage(rows: int, cols: int, obs_prob = 0.2):

    # generate obstacles with obs_prob probability
    num_obstacles = int(rows * cols * obs_prob)

    stage = np.full((rows, cols), 0)

    # Set 1s at random positions for the specified percentage
    indices = np.random.choice(rows * cols, num_obstacles, replace=False)
    stage.flat[indices] = 1

    return stage

def generate_agents(real_stage, num_agents: int = 1):
    agents = []

    if num_agents <= 0:
        num_agents = 1

    zero_coordinates = list(zip(*np.where(real_stage == 0)))

    # Create the "explored" stage
    for _ in range(num_agents):
        if zero_coordinates:
            start = random.choice(zero_coordinates)
            zero_coordinates.remove(start)
            goal = random.choice(zero_coordinates)
            agents.append(Agent((start[0], start[1]), (goal[0], goal[1]), real_stage))
        else:
            break

    return agents

def concat_exp_stage(agents):
    conc_expl_stage = agents[0].explored_stage.copy()
    for a in agents[1:]:
        conc_expl_stage[conc_expl_stage == -1] = a.explored_stage[conc_expl_stage == -1]

    # conc_expl_stage[conc_expl_stage == 2] = 0

    num_minus_1 = np.sum(conc_expl_stage == -1)
    explored_percentage = 1 - (num_minus_1 / (conc_expl_stage.shape[0] * conc_expl_stage.shape[1]))

    return conc_expl_stage, explored_percentage
