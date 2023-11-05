import copy
import random
import numpy as np
from classes.stage import *
from algorithms.astar import a_star

# Stage shape:
rows, cols = 10, 10
obs_prob = 0.2

# Number of agents:
num_agents = 3

max_episodes = 100
# directions = ['up', 'down', 'left', 'right']

grid = generate_stage(rows, cols, obs_prob)

agents = generate_agents(real_stage = grid, num_agents = num_agents)
all_agents = copy.deepcopy(agents)

for agent in agents:
    grid[agent.x, agent.y] = 2  # Mark initial agent positions

episode = 0

print(concat_exp_stage(agents))
print(grid)

while any((agent.x, agent.y) != agent.goal for agent in agents) and episode < max_episodes:
    episode += 1
    agent_count = 1
    for agent in agents:
        # agent.agent_view()  # update agent view
        if (agent.x, agent.y) == agent.goal:
            grid[agent.x, agent.y] = 0
            agents.remove(agent)
            continue  # Agent has reached its goal

        path = a_star(grid, (agent.x, agent.y), agent.goal)
        print(f"Agent{agent_count} Path: {path}")
        print(f"Agent{agent_count} Goal: {agent.goal}")
        agent_count += 1
        if path:
            grid[agent.x, agent.y] = 0  # Mark the old position as unoccupied
            agent.x, agent.y = path[1]  # Update agent position
            agent.agent_view()
            grid[agent.x, agent.y] = 2  # Mark the new position as occupied by agent

    # print(f"{len(agents)} agents remaining.")
    print(grid)


print(concat_exp_stage(all_agents))