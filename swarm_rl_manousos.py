# -*- coding: utf-8 -*-
import os
import sys
import time
import copy
import random
import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from IPython import get_ipython
from scipy.spatial.distance import cdist
from PIL import Image
import re

from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def check_run_colab() -> bool:
    return bool('google.colab' in sys.modules)

"""Mount Google Drive:"""

# Check if the current environment is Google Colab
if check_run_colab():
    print("Running in Google Colab!")
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print("Not running in Google Colab.")

"""The Agent Class:"""

class Agent:

  def __init__(self, start: tuple, goal: tuple, real_stage, view_range=2):
      self.x = start[0]
      self.y = start[1]
      self.goal = goal
      self.view_range = view_range
      self.explored_stage = np.full_like(real_stage, -1)
      self.explored_stage[self.x, self.y] = 0
      self.agent_view(real_stage)
      self.start_time = time.time()
      # voronoi related parameters:
      self.voronoi_coords = None
      # self.broadcast_range = max(real_stage.shape[0], real_stage.shape[1]) *2 #// 4
      self.broadcast_range = max(real_stage.shape[0], real_stage.shape[1]) // 4


  def agent_view(self, real_stage):
    """ Refreshes the explored map of the agent (sees up, down, left, right). """
    up_obs, upleft_obs, upright_obs, down_obs, downleft_obs, downright_obs, left_obs, right_obs = False, False, False, False, False, False, False, False
    for i in range(self.view_range):
      if self.x > i:  # checks up
        tmp_x = self.x - i - 1
        if not up_obs:  # stops if it sees obstacle
          self.explored_stage[(tmp_x, self.y)] = real_stage[(tmp_x, self.y)]
          if real_stage[(tmp_x, self.y)]:
            up_obs = True
        if self.y > i and not upleft_obs:  # up-left
          if not upleft_obs:  # stops if it sees obstacle
            self.explored_stage[(tmp_x, self.y - i - 1)] = real_stage[(tmp_x, self.y - i - 1)]
            if real_stage[(tmp_x, self.y - i - 1)]:
              upleft_obs = True
        if self.y < len(real_stage[0]) - i - 1: # up-right
          if not upright_obs:  # stops if it sees obstacle
            self.explored_stage[(tmp_x, self.y + i + 1)] = real_stage[(tmp_x, self.y + i + 1)]
            if real_stage[(tmp_x, self.y + i + 1)]:
              upright_obs = True

      if self.x < len(real_stage) - i - 1:  # checks down:
        tmp_x = self.x + i + 1
        if not down_obs:
          self.explored_stage[(tmp_x, self.y)] = real_stage[(tmp_x, self.y)]
          if real_stage[(tmp_x, self.y)]:
            down_obs = True
        if self.y > i:  # down-left
          if not downleft_obs:
            self.explored_stage[(tmp_x, self.y - i - 1)] = real_stage[(tmp_x, self.y - i - 1)]
            if real_stage[(tmp_x, self.y - i - 1)]:
              downleft_obs = True
        if self.y < len(real_stage[0]) - i - 1: # down-right
          if not downright_obs:
            self.explored_stage[(tmp_x, self.y + i + 1)] = real_stage[(tmp_x, self.y + i + 1)]
            if real_stage[(tmp_x, self.y + i + 1)]:
              downright_obs = True

      if self.y > i and not left_obs:  # left (& stops if it sees obstacle)
        self.explored_stage[(self.x, self.y - i - 1)] = real_stage[(self.x, self.y - i - 1)]
        if real_stage[(self.x, self.y - i - 1)]:
          left_obs = True

      if self.y < len(real_stage[0]) - i - 1 and not right_obs: # right (& stops if it sees obstacle)
        self.explored_stage[(self.x, self.y + i + 1)] = real_stage[(self.x, self.y + i + 1)]
        if real_stage[(self.x, self.y + i + 1)]:
          right_obs = True

    self.explored_stage[self.explored_stage == 2] = 0

  def check_goal(self):
    if (self.x, self.y) == self.goal:
      return True
    return False

"""A* Algorithm (source [here](https://pypi.org/project/python-astar/)):"""

"""
Python-astar - A* path search algorithm
"""

class Tile:
    """A tile is a walkable space on a map."""
    distance = 0
    came_from = None

    def __init__(self, x, y, weight=1):
        self.x = x
        self.y = y
        self.weight = 1
        assert (self.x is not None and self.y is not None)

    def update_origin(self, came_from):
        """Update which tile this one came from."""
        self.came_from = came_from
        self.distance = came_from.distance + self.weight

    def __eq__(self, other):
        """A tile is the same if they have the same position"""
        return (other and self.x == other.x and self.y == other.y)

    def __lt__(self, other):
        """We want the shortest distance tile to find the happy path.
        This is used by min() so we can just compare them :)
        """
        return (self.distance + self.weight <= other.distance)

    def __hash__(self):
        """We need this so we can use a set()"""
        return hash(str(self))

    @property
    def pos(self):
        """a (x, y) tuple with position on the grid"""
        return (self.x, self.y)

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return str(self)


class AStar:
    """The A Star (A*) path search algorithm"""

    def __init__(self, world, coverage_mode: bool = False):
        world2 = copy.deepcopy(world)
        world2[world2 == -1] = 0
        if coverage_mode: # coverage_mode == different goals.
          world2[world2 == 2] = 1 # astar takes agents into account.
        else:
          world2[world2 == 2] = 0
        self.world = world2

    def search(self, start_pos, target_pos):
        """A_Star (A*) path search algorithm"""
        start = Tile(*start_pos)
        self.open_tiles = set([start])
        self.closed_tiles = set()

        # while we still have tiles to search
        while len(self.open_tiles) > 0:
            # get the tile with the shortest distance
            tile = min(self.open_tiles)
            # check if we're there. Happy path!
            if tile.pos == target_pos:
                return self.rebuild_path(tile)
            # search new ways in the neighbor's tiles.
            self.search_for_tiles(tile)

            self.close_tile(tile)
        # if we got here, path is blocked :(
        return None

    def search_for_tiles(self, current):
        """Search for new tiles in the maze"""
        for other in self.get_neighbors(current):
            if self.is_new_tile(other):
                other.update_origin(current)
                self.open_tiles.add(other)

            # if this other has gone a farthest distance before
            #   then we just found a new and shortest way to it.
            elif other > current:
                other.update_origin(current)
                if other in self.closed_tiles:
                    self.reopen_tile(other)

    def get_neighbors(self, tile):
        """Return a list of available tiles around a given tile"""
        min_x = max(0, tile.x - 1)
        max_x = min(len(self.world)-1, tile.x + 1)
        min_y = max(0, tile.y - 1)
        max_y = min(len(self.world[tile.x])-1, tile.y + 1)

        available_tiles = [
            (min_x, tile.y),
            (max_x, tile.y),
            (tile.x, min_y),
            (tile.x, max_y),
        ]
        neighbors = []
        for x, y in available_tiles:
            if (x, y) == tile.pos:
                continue

            if self.world[x][y] == 0:
                neighbors.append(Tile(x, y))

        return neighbors

    def rebuild_path(self, current):
        """Rebuild the path from each tile"""
        self.last_tile = current
        path = []
        while current is not None:
            path.append(current)
            current = current.came_from
        path.reverse()
        # return a list with tuples
        return [tile.pos for tile in path]

    def is_new_tile(self, tile):
        """Check if this is a proviously unknown tile"""
        return (
            tile not in self.open_tiles
            and tile not in self.closed_tiles
        )

    def reopen_tile(self, tile):
        """Reinstate a tile in the open list"""
        self.closed_tiles.remove(tile)
        self.open_tiles.add(tile)

    def close_tile(self, tile):
        """Remove tile from open_tiles, as we are done testing it"""
        self.open_tiles.remove(tile)
        self.closed_tiles.add(tile)

def draw_maze(maze, path=None, goal=None, save_gif=False):
    fig, ax = plt.subplots(figsize=(10,10))

    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')

    if path is not None:
        x_coords = [x[1] for x in path]
        y_coords = [y[0] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)
        ax.scatter(path[-1][1], path[-1][0], color='red', s=100, marker='s')

    if goal is not None:
        ax.scatter(goal[1], goal[0], color='red', s=100, marker='s')

    ax.set_xticks([])
    ax.set_yticks([])

    # Saves images to folder ========================
    if save_gif:
        if not os.path.exists("tmp_img"):
            os.makedirs("tmp_img")
        filepath = os.path.join("tmp_img", f"{time.time()}.png")
        plt.savefig(filepath)
    # =============================================

    plt.show()

from matplotlib.colors import ListedColormap, Normalize

def draw_maze_voronoi(v_map, agent_explored=None, path=None, goal=None, save_gif=False):
    unique_values = np.unique(v_map)
    # print(len(unique_values))
    if len(unique_values) > 11:
        print("Colored plot not supported for more than 10 agents. Plotting in grayscale:")
        draw_maze(v_map, path, goal, save_gif) if agent_explored is None else draw_maze(agent_explored, path, goal, save_gif)
        return

    tmp_v_map = copy.deepcopy(v_map)
    fig, ax = plt.subplots(figsize=(10,10))

    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    if agent_explored is not None:
        tmp_v_map[agent_explored == 1] = -1  # obstacles -> gray
        tmp_v_map[agent_explored == -1] = -2    # unexplored areas -> white
        tmp_v_map[agent_explored == 2] = -3 # agents -> black
    color_dict = {-3: 'black', -2: 'white', -1: '#555555', 0: "blue", 1: "brown", 2: "green", 3: "purple", 4: "orange", 5: "cyan", 6: "magenta", 7: "yellow", 8: "lime", 9: "pink"}
    # else:
    #     color_dict = {-1: 'darkgray', 0: "blue", 1: "brown", 2: "green", 3: "purple", 4: "orange", 5: "cyan", 6: "magenta", 7: "yellow", 8: "lime", 9: "pink"}

    colors = [color_dict[val] for val in sorted(color_dict.keys())]
    cmap = ListedColormap(colors)

    norm = Normalize(vmin=min(color_dict.keys()), vmax=max(color_dict.keys()))

    plt.imshow(tmp_v_map, cmap=cmap, norm=norm, interpolation='nearest')
    # if agent_explored is None:
    #     plt.colorbar()

    if path is not None:
        x_coords = [x[1] for x in path]
        y_coords = [y[0] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)
        ax.scatter(path[-1][1], path[-1][0], color='red', s=100, marker='s')

    if goal is not None:
        ax.scatter(goal[1], goal[0], color='red', s=100, marker='s')

    ax.set_xticks([])
    ax.set_yticks([])

    # Saves images to folder ========================
    if save_gif:
        if not os.path.exists("tmp_img"):
            os.makedirs("tmp_img")
        filepath = os.path.join("tmp_img", f"{time.time()}.png")
        plt.savefig(filepath)
    # =============================================

    plt.show()
    # print("---------------")

def images_to_gif(gif_filename=f"maze_{time.time()}.gif", duration=300, image_folder="tmp_img", gif_folder="utils"):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') and os.path.isfile(os.path.join(image_folder, f))]

    image_files.sort()

    images = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        images.append(image)

    gif_filepath = os.path.join(gif_folder, gif_filename)
    images[0].save(gif_filepath, save_all=True, append_images=images[1:], loop=0, duration=duration)

    for image_file in image_files:
        os.remove(os.path.join(image_folder, image_file))


def generate_stage(rows: int, cols: int, obs_prob = 0.2):

  # generate obstacles with obs_prob probability
  num_obstacles = int(rows * cols * obs_prob)

  stage = np.full((rows, cols), 0)

  # Set 1s at random positions for the specified percentage
  indices = np.random.choice(rows * cols, num_obstacles, replace=False)
  stage.flat[indices] = 1

  return stage

def create_maze(rows, cols, obs_prob=0.8):
    rows = int(rows / 2)
    cols = int(cols / 2)

    maze = np.ones((rows*2+1, cols*2+1))

    x, y = (0, 0)

    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < rows and ny < cols and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    zero_indices = np.argwhere(maze == 0)
    zero_coords = [tuple(index) for index in zero_indices]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # adds randomly crosses of free space.
    for z in zero_coords:
       if random.random() >= obs_prob:
          for dx, dy in directions:
            nx, ny = z[0] + dx, z[1] + dy
            maze[nx, ny] = 0

    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # removes crosses (so agents wont be stuck).
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            walls = []
            for d in directions:
                neighbor_i = i + d[0]
                neighbor_j = j + d[1]
                # Check if neighbor is in bounds
                if 0 <= neighbor_i < maze.shape[0] and 0 <= neighbor_j < maze.shape[1] and maze[(neighbor_i, neighbor_j)]:
                    walls.append((neighbor_i, neighbor_j))
            if len(walls) >= len(directions):
                for coord in walls:
                    maze[coord] = 0

    # re-adds the boundaries (after cross removed).
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1
    # draw_maze(maze)

    return maze


def generate_agents(real_stage, num_agents: int = 1, view_range: int = 2, coverage_mode: bool = False):

  agents = []

  if num_agents <= 0:
    num_agents = 1

  zero_coordinates = list(zip(*np.where(real_stage == 0)))
  goal = random.choice(zero_coordinates)
  zero_coordinates.remove(goal)
  # Create the "explored" stage
  for _ in range(num_agents):
    if zero_coordinates:
      start = random.choice(zero_coordinates)
      zero_coordinates.remove(start)
      agents.append(Agent((start[0], start[1]), (goal[0], goal[1]), real_stage, view_range))
      if coverage_mode: # puts different goals
        goal = random.choice(zero_coordinates)
        zero_coordinates.remove(goal)
    else:
      break

  return agents


def update_total_explored(agents, coverage_mode=False, voronoi_mode=False):
  if len(agents) == 0:
    return

  total_explored = np.full(agents[0].explored_stage.shape, -1)

  # fills the border with 1s
  total_explored[0, :] = 1
  total_explored[-1, :] = 1
  total_explored[:, 0] = 1
  total_explored[:, -1] = 1

  for a in agents:
    total_explored[total_explored == -1] = a.explored_stage[total_explored == -1]
    total_explored[total_explored == 2] = 0

  for a in agents:
    if coverage_mode: # agents never leave the stage when coverage mode.
      total_explored[a.x, a.y] = 2
    elif not a.check_goal():  # if agent has reached goal -> collidable (as if it is removed from the stage).
      total_explored[a.x, a.y] = 2

  if not voronoi_mode:  # if voronoi mode, DOES NOT share information with other agents.
    # Total explored contains the concats of all agents stages:
    for a in agents:
      a.explored_stage = copy.deepcopy(total_explored)

  return total_explored

def find_unexp_voronoi(agent):
    # return unexpl_coords  # this works fine.

    unexp_vor_coords = []
    for v_coord in list(agent.voronoi_coords):
        # print(type(v_coord))
        if v_coord != (agent.x, agent.y) and agent.explored_stage[v_coord] == -1:
            unexp_vor_coords.append(v_coord)

    ## debug:
    # # print(unexp_vor_coords)
    # debug_matrix = np.full_like(agent.explored_stage, 0)
    # for v in unexp_vor_coords:
    #     debug_matrix[v] = 1
    # print("Unexplored voronoi coords:")
    # draw_maze(debug_matrix)

    return np.array(unexp_vor_coords)


def calculate_expl_percentage(total_explored):
  subarray = total_explored[1:-1, 1:-1] # gets all rows and columns except from the borders (agents already know they are obstacles).
  num_minus_1 = np.sum(subarray == -1)  # gets the unexplored areas.
  explored_percentage = 1 - (num_minus_1 / (subarray.shape[0] * subarray.shape[1]))
  return explored_percentage  # equals to M / P of paper.


def check_real_expl(real_grid, total_explored, debug=False):
  total_explored = np.where(total_explored == -1, 1, total_explored)  # converts all unexplored to 1s.
  expl_perc = calculate_expl_percentage(total_explored)
  # the borders of the maze are removed from the above calculation (bc the agents already know they are obstacles,
  # thats why we dont calculate them). To continue in the same logic, we also remove borders below.

  real = copy.deepcopy(real_grid[1:-1, 1:-1])
  real = np.where(real == 2, 0, real)
  tot = copy.deepcopy(total_explored[1:-1, 1:-1])
  tot = np.where(tot == 2, 0, tot)
  tot = np.where(tot == -1, 1, tot)

  false_positions = np.where(np.not_equal(real, tot))[0]
  total_false = np.sum(np.not_equal(real, tot)) / (real_grid.shape[0] * real_grid.shape[1])

  if debug: # print statements:
    if total_false == 0:
      print("Real == Explored? TRUE")
    else:
      print("Real == Explored? FALSE")
      print("Positions of False values:", false_positions)
      draw_maze(real)
      draw_maze(tot)
  return false_positions, expl_perc - total_false

def calc_exploration_efficiency(total_explored, sum_d):
    tot = copy.deepcopy(total_explored[1:-1, 1:-1])
    tot = np.where(tot == 2, 0, tot)
    tot = np.where(tot == -1, 1, tot)
    return np.count_nonzero(tot >= 0) / sum_d

def move_astar(start_grid, start_agents, debug=True):

  grid = copy.deepcopy(start_grid)

  agents = copy.deepcopy(start_agents)

  for agent in agents:
      grid[agent.x, agent.y] = 2  # Mark initial agent positions
  total_explored = update_total_explored(agents)
  total_agents = len(agents)
  # print(total_agents)
  avg_eps_time = []
  avg_rounds = []
  num_finish = 0

  rounds = 0
  while any((agent.x, agent.y) != agent.goal for agent in agents):
      rounds+=1
      eps_start_time = time.time()
      path_none = 0
      for agent in agents:
          if agent.check_goal():
              num_finish += 1
              total_explored = update_total_explored(agents)
              agents.remove(agent)
              avg_rounds.append(rounds)
              continue  # Agent has reached its goal

          path = AStar(agent.explored_stage).search((agent.x, agent.y), agent.goal)
          if debug:
            draw_maze(agent.explored_stage, path=path)

          if path:
              if agent.explored_stage[path[1]] != 2:
                grid[agent.x, agent.y] = 0  # Mark the old position as unoccupied
                agent.x, agent.y = path[1]  # Update agent position
                grid[agent.x, agent.y] = 2  # Mark the new position as occupied by agent
                # if agent.check_goal():
          else:
            path_none += 1
          agent.agent_view(start_grid)
          total_explored = update_total_explored(agents)
      avg_eps_time.append(time.time() - eps_start_time)
      if path_none >= len(agents): # stops if no agents have moved.
          print(path_none, len(agents))
          print("STOP PATH NONE")
          break

  for agent in agents:  # gets the time of some finished agents (that have not been counted),
    if agent.check_goal():
      num_finish += 1
      avg_rounds.append(rounds)

  return calculate_expl_percentage(total_explored), rounds, np.mean(avg_rounds), np.mean(avg_eps_time), num_finish / total_agents, total_explored


def nearest_frontier(x, y, unexpl_coords, explored_stage) -> tuple:
    """Returns the new goal according to nearest frontier."""
    min_path = np.inf
    min_coord = (x, y)
    for u_c in unexpl_coords:
      path = AStar(explored_stage, coverage_mode=True).search((x, y), tuple(u_c))
      if path is not None and len(path) <= min_path:
        min_path = len(path)
        min_coord = tuple(u_c)
    # print(f"Length: {min_path} || u_coords length: {len(unexpl_coords)}")
    return min_coord

def calculate_sum_aij(i, j, explored_stage, u_r=None):
    neighbors_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    sum_neighbors = 0

    for offset in neighbors_offsets:
        neighbor_i = i + offset[0]
        neighbor_j = j + offset[1]
        # Check if neighbor is in bounds
        if 0 <= neighbor_i < explored_stage.shape[0] and 0 <= neighbor_j < explored_stage.shape[1]:
            a_ij = 1 if explored_stage[neighbor_i, neighbor_j] <= 0 else 0 # 1 if no obstacle, else 0
            sum_neighbors += a_ij * u_r[(neighbor_i, neighbor_j)] if u_r is not None else a_ij
    return sum_neighbors

def calculate_s(explored_stage):
    """Calculates the entire s matrix, which in each cell contains corresponding s(xij)."""
    c = copy.deepcopy(explored_stage)
    c[c >= 0] = 1  # explored -> 1
    c[c == -1] = 0  # not explored -> 0

    S_t = np.sum(c) # sums the explored (1) indexes.

    s = np.zeros((explored_stage.shape[0], explored_stage.shape[1]))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[(i, j)] = max(0, 1 - c[(i, j)]) * S_t
    return s

def calculate_uij(i, j, explored_stage, alpha, u_r, s_ij):
    """Calculates next iteration (r+1) u_ij"""
    t1 = calculate_sum_aij(i, j, explored_stage, u_r) + s_ij
    return t1 / (calculate_sum_aij(i, j, explored_stage) + alpha)

def update_attracting_field(u, alpha, s, explored_stage):
    """Function to update the attracting field iteratively"""
    new_u = np.zeros_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            new_u[(i, j)] = calculate_uij(i, j, explored_stage, alpha, u, s[(i, j)])
    return new_u

def calc_attractive_field(explored_stage, alpha=0.1, max_iter=100):
    """Calculates the entire attractive field matrix for a specified explored stage."""
    u = np.zeros((explored_stage.shape[0], explored_stage.shape[1]))  # Initial attracting field
    # Iteratively update the attracting field
    s = calculate_s(explored_stage)
    for _ in range(max_iter):
        new_u = update_attracting_field(u, alpha, s, explored_stage)
        u = new_u
    return u

def get_next_position(u, agent):
	# Updated next pos for agent as the non ocuppied neighbouring nodes where is the largest value of the scalar field u or current position if all neigbouring nodes are ocuppied;
    i, j = agent.x, agent.y
    L = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Neighboring nodes
    max_v = -np.inf
    target_pos = (i, j)
    for l in L:
        x, y = i + l[0], j + l[1]
        if 0 <= x < u.shape[0] and 0 <= y < u.shape[1]:
            if agent.explored_stage[x, y] <= 0 and u[x, y] > max_v:
                target_pos = (x, y)
                max_v = u[x, y]
    return target_pos

def move_hedac_coverage(agents, start_grid, coverage_finish = 1.0, alpha=10, max_iter=100, debug=False, save_images=False):
    visited_nodes = set()

    agents = copy.deepcopy(agents)

    grid = copy.deepcopy(start_grid)

    total_explored = update_total_explored(agents, True)

    avg_eps_time = []
    rounds = 0
    count_same = 0
    sum_dist = 0

    while calculate_expl_percentage(total_explored) < coverage_finish:
        old_grid = copy.deepcopy(grid)
        rounds += 1
        eps_start_time = time.time()
        u = calc_attractive_field(total_explored, alpha, max_iter)  # A function to update and solve the linear system
        for i in agents:
            next_position = get_next_position(u, i)  # A function to get next position for agent i
            if debug:
                draw_maze(i.explored_stage, goal=next_position, save_gif=save_images)
            if (i.x, i.y) != next_position:
                sum_dist += 1
            grid[i.x, i.y] = 0  # Mark the old position as unoccupied
            i.x, i.y = next_position  # Update agent position
            grid[i.x, i.y] = 2  # Mark the new position as occupied by agent
            visited_nodes.add(next_position)
            i.agent_view(start_grid)
            total_explored = update_total_explored(agents, True)    # A function to exchange information about discovered nodes between agents
        avg_eps_time.append(time.time() - eps_start_time)
        if np.all(old_grid == grid):    # checks if agents are stuck (does not happen, but just in case)
            count_same += 1
            if count_same >= int(max_iter / 10):
                break

    re = check_real_expl(start_grid, total_explored)  # gets the difference of explored & real grid

    return re[1], rounds, total_explored, np.mean(avg_eps_time), sum_dist, calc_exploration_efficiency(total_explored, sum_dist)

def cost_utility_mnm(x, y, unexpl_coords, explored_stage, agents) -> tuple:
    """Returns the new goal according to mnm cost utility."""
    min_path = np.inf
    min_path_dict = {}
    target_coord = (x, y)
    for u_c in unexpl_coords:
      path = AStar(explored_stage, coverage_mode=True).search((x, y), tuple(u_c))
      if path is not None and len(path) <= min_path:
        min_path = len(path)
        target_coord = tuple(u_c)
        min_path_dict[tuple(u_c)] = min_path

    # creates a new dict with keys the coords which have the shortest path.
    min_min_path_dict = {k: min_path for k, v in min_path_dict.items() if v == min_path}

    max_util = -1
    for k in min_min_path_dict:
        utility = 0
        for a in agents:
          if a.x == x and a.y == y: #
             continue
          utility += abs(k[0] - a.x) + abs(k[1] - a.y)  # manhattan
        if utility > max_util:
          max_util = utility
          target_coord = k
    # print("Cost utility")

    return target_coord

def calculate_utility_jgr(x, y, view_range, explored_stage):
  up_obs, upleft_obs, upright_obs, down_obs, downleft_obs, downright_obs, left_obs, right_obs = False, False, False, False, False, False, False, False
  sum_unex = 0
  for i in range(view_range):
    if x > i:  # checks up
      tmp_x = x - i - 1
      if not up_obs:  # stops if it sees obstacle
        if explored_stage[(tmp_x, y)] == -1:
          sum_unex += 1
        elif explored_stage[(tmp_x, y)] > 0:
          up_obs = True
      if y > i:  # up-left
        if not upleft_obs:  # stops if it sees obstacle
          if explored_stage[(tmp_x, y - i - 1)] == -1:
            sum_unex += 1
          elif explored_stage[(tmp_x, y - i - 1)] > 0:
            upleft_obs = True
      if y < len(explored_stage[0]) - i - 1: # up-right
        if not upright_obs:  # stops if it sees obstacle
          if explored_stage[(tmp_x, y + i + 1)] == -1:
            sum_unex += 1
          elif explored_stage[(tmp_x, y + i + 1)] > 0:
            upright_obs = True

    if x < len(explored_stage) - i - 1:  # checks down:
      tmp_x = x + i + 1
      if not down_obs:
        if explored_stage[(tmp_x, y)] == -1:
          sum_unex += 1
        elif explored_stage[(tmp_x, y)] > 0:
          down_obs = True
      if y > i:  # down-left
        if not downleft_obs:
          if explored_stage[(tmp_x, y - i - 1)] == -1:
            sum_unex += 1
          elif explored_stage[(tmp_x, y - i - 1)] > 0:
            downleft_obs = True
      if y < len(explored_stage[0]) - i - 1: # down-right
        if not downright_obs:
          if explored_stage[(tmp_x, y + i + 1)] == -1:
            sum_unex += 1
          elif explored_stage[(tmp_x, y + i + 1)] > 0:
            downright_obs = True

    if y > i and not left_obs:  # left (& stops if it sees obstacle)
      if explored_stage[(x, y - i - 1)] == -1:
        sum_unex += 1
      elif explored_stage[(x, y - i - 1)] > 0:
        left_obs = True

    if y < len(explored_stage[0]) - i - 1 and not right_obs: # right (& stops if it sees obstacle)
      if explored_stage[(x, y + i + 1)] == -1:
        sum_unex += 1
      elif explored_stage[(x, y + i + 1)] > 0:
        right_obs = True

  return sum_unex / (np.pi * (view_range ** 2))

def cost_utility_jgr(x, y, unexpl_coords, explored_stage, agent_view, lambda_=0.8) -> tuple:
    L = {}
    max_path = -1
    for u_c in unexpl_coords:
      path = AStar(explored_stage, coverage_mode=True).search((x, y), tuple(u_c))
      if path is not None:
        L[tuple(u_c)] = len(path)
        if len(path) > max_path:
          max_path = len(path)

    max_c = (x, y)
    max_u = -1
    for d in L:
      tmp = calculate_utility_jgr(d[0], d[1], agent_view, explored_stage) - (lambda_*(L[d] / max_path))
      if tmp > max_u:
        max_u = tmp
        max_c = d
    return max_c

def compute_bso_cost_matrix(agents, frontiers, explored_stage):
    cost_matrix = np.full((len(agents), len(frontiers)), np.inf)

    for i, a in enumerate(agents):
        for j, frontier in enumerate(frontiers):
            path = AStar(explored_stage, coverage_mode=True).search((a.x, a.y), tuple(frontier))
            if path is not None:
                cost_matrix[i][j] = len(path)

    return cost_matrix

def cost_utility_bso(agents, frontiers, explored_stage):
    cost_matrix = compute_bso_cost_matrix(agents, frontiers, explored_stage)
    p_matrix = np.zeros((len(agents), len(frontiers)))

    for i, a in enumerate(agents):
        for j, _ in enumerate(frontiers):
            for k, _ in enumerate(agents):
                if k != i:
                    p_matrix[i][j] += 1 if cost_matrix[k][j] < cost_matrix[i][j] else 0

        min_indices = np.where(p_matrix[i] == np.min(p_matrix[i]))[0]

        if len(min_indices) == 1:
            a.goal = tuple(frontiers[min_indices[0]])
        else:
            min_cost = np.inf
            for j in min_indices:
                if cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    a.goal = tuple(frontiers[j])

def calc_umnm(a, agents):
  utility = 0
  for i in agents:
    if i.x == a[0] and i.y == a[1]: # ignores
        continue
    utility += abs(a[0] - i.x) + abs(a[1] - i.y)  # manhattan
  return utility

def calc_ujgr(a, Rs, explored_stage):
  return calculate_utility_jgr(a[0], a[1], Rs, explored_stage) * (np.pi * (Rs**2))

def norm_values(u):
  min_v = np.min(u)
  max_v = np.max(u)
  return (u - min_v) / (max_v - min_v)

def calc_new_util_path(x, y, agents, total_explored, Rs=2, hedac=False, min_min_path_dict=None):
  u_mnm = np.full_like(total_explored, 0)
  u_jgr = np.full_like(total_explored, 0)

  for u_c in min_min_path_dict:
    u_mnm[tuple(u_c)] = calc_umnm(tuple(u_c), agents)

    # calculate u_jgr
    path = AStar(total_explored, coverage_mode=True).search((x, y), tuple(u_c))
    sum_ujr = 0
    for coord in path:
      sum_ujr += calc_ujgr(tuple(coord), Rs, total_explored)
    u_jgr[tuple(u_c)] = sum_ujr
  u_mnm = norm_values(u_mnm)
  u_jgr = norm_values(u_jgr)

  if hedac:
    u_hedac = norm_values(calc_attractive_field(total_explored, alpha=10))
    return u_mnm + u_jgr + u_hedac

  return u_mnm + u_jgr


def calc_new_util(agents, total_explored, Rs=2, hedac=False):
  u_mnm = np.full_like(total_explored, 0)
  for i in range(u_mnm.shape[0]):
      for j in range(u_mnm.shape[1]):
          u_mnm[(i, j)] = calc_umnm((i, j), agents)
  u_mnm = norm_values(u_mnm)

  u_jgr = np.full_like(total_explored, 0)
  for i in range(u_jgr.shape[0]):
      for j in range(u_jgr.shape[1]):
          u_jgr[(i, j)] = calc_ujgr((i, j), Rs, total_explored)
  u_jgr = norm_values(u_jgr)

  if hedac:
    u_hedac = norm_values(calc_attractive_field(total_explored, alpha=10))
    return u_mnm + u_jgr + u_hedac

  return u_mnm + u_jgr


def cost_utility_new(x, y, unexpl_coords, explored_stage, agents, view_range=2, algo=None) -> tuple:
    hedac = False
    if algo == 'new_cu_hedac_diffgoal' or algo == 'new_cu_hedac_diffgoal_path' or algo == 'new_cu_hedac_same':
       hedac = True

    min_path = np.inf
    min_path_dict = {}
    target_coord = (x, y)
    # print(f"Goals left: {unexpl_coords}")
    # print("================")
    for u_c in unexpl_coords:
      path = AStar(explored_stage, coverage_mode=True).search((x, y), tuple(u_c))
      if path is not None and len(path) <= min_path:
        min_path = len(path)
        target_coord = tuple(u_c)
        min_path_dict[tuple(u_c)] = min_path

    # creates a new dict with keys the coords which have the shortest path.
    min_min_path_dict = {k: min_path for k, v in min_path_dict.items() if v == min_path}
    # print(len(min_min_path_dict))
    if algo == 'new_cu_diffgoal_path' or algo == 'new_cu_hedac_diffgoal_path':
      utility = calc_new_util_path(x, y, agents, explored_stage, view_range, hedac, min_min_path_dict)
    else:
      utility = calc_new_util(agents, explored_stage, view_range, hedac)

    max_util = -1
    for k in min_min_path_dict:
        if utility[k] > max_util:
          max_util = utility[k]
          target_coord = k

    # print(target_coord)
    return target_coord

def update_goals_new_cost_util(agents, unexpl_coords, total_explored, start=False, algo='new_cu_diffgoal'):
  un_coords = copy.deepcopy(unexpl_coords.tolist())
  for a in agents:
    if total_explored[a.goal] == -1 and list(a.goal) in un_coords and len(unexpl_coords) >= len(agents):
      un_coords.remove(list(a.goal))  # agent a is exploring the goal, and will continue to explore it.

  for a in agents:
    if not start and total_explored[a.goal] == -1:
        continue
    a.goal = cost_utility_new(a.x, a.y, un_coords, a.explored_stage, agents, a.view_range, algo=algo)
    if a.goal != (a.x, a.y) and len(unexpl_coords) >= len(agents):
      un_coords.remove(list(a.goal))
      # print(f"Goal Removed: {list(a.goal)}")
      # print(f"Goals left: {un_coords}")
      # print("================")

def update_goals_new_cost_util_voronoi(agents, unexpl_coords, start=False, algo='new_cu_diffgoal'):
  un_coords = []
  goals = {}
  un_coords_all = copy.deepcopy(unexpl_coords.tolist())
  for a in agents:
    un_coords.append(find_unexp_voronoi(a).tolist())

  # test_i = (-1, -1)
  for a_i, a in enumerate(agents):
    if not start and a.explored_stage[a.goal] == -1:
        # goals.update({a.goal: 1}) if a.goal not in goals else goals.pop(a.goal)
        # test_i = a.goal
        continue
    a.goal = cost_utility_new(a.x, a.y, un_coords[a_i], a.explored_stage, agents, a.view_range, algo=algo)
    if a.goal not in goals:
      goals[a.goal] = 1
    elif a.goal != (a.x, a.y) and len(un_coords_all) >= len(agents):
      while a.goal in goals and len(un_coords[a_i]) > 1:
        un_coords[a_i].remove(list(a.goal))
        a.goal = cost_utility_new(a.x, a.y, un_coords[a_i], a.explored_stage, agents, a.view_range, algo=algo)
      if a.goal not in goals:
        goals[a.goal] = 1

  goals = {}
  for a_i, a in enumerate(agents):
    if a.goal in goals and len(un_coords[a_i]) > 1 and len(un_coords_all) >= len(agents):
      goals[a.goal] += 1
      print(un_coords)
      print(f"Same goal {a.goal} appeared {goals[a.goal]} times.")# {test_i}")
    else:
      goals[a.goal] = 1


def update_goals(agents, total_explored, start=False, algo='nf', lambda_=0.8, voronoi_mode=False):
  """ Function to update the goals of the agents (if they are explored). """
  unexpl_coords = np.argwhere(total_explored == -1)
  if len(unexpl_coords) <= 0:
    return

  diffgoal_algos = ['new_cu_diffgoal', 'new_cu_hedac_diffgoal', 'new_cu_diffgoal_path', 'new_cu_hedac_diffgoal_path']

  if algo in diffgoal_algos:
    if voronoi_mode:
      update_goals_new_cost_util_voronoi(agents, unexpl_coords, start, algo=algo)
      # update_goals_new_cost_util_voronoi(agents, unexpl_coords, total_explored, v_map, start, algo=algo,)
    else:
      update_goals_new_cost_util(agents, unexpl_coords, total_explored, start, algo=algo)
    return

  if algo == 'cu_bso':
    if voronoi_mode:
      raise NotImplementedError(f"{algo} has not been implemented in voronoi mode.")
    cost_utility_bso(agents, unexpl_coords, total_explored)
    return

  for a_i, a in enumerate(agents):
    # if not start and total_explored[a.goal] == -1 and not r:
    if not start and total_explored[a.goal] == -1:
      continue

    # New impl ============================
    if algo == 'nf':  # nearest frontier
      if not voronoi_mode:
        a.goal = nearest_frontier(a.x, a.y, unexpl_coords, a.explored_stage)
      else:
        # print(f"Agent {a_i} path length & unexpl coords length:")
        a.goal = nearest_frontier(a.x, a.y, find_unexp_voronoi(a), a.explored_stage)
    elif algo == 'cu_mnm':  # cost utility - mnm
      if not voronoi_mode:
        a.goal = cost_utility_mnm(a.x, a.y, unexpl_coords, a.explored_stage, agents)
      else:
        a.goal = cost_utility_mnm(a.x, a.y, find_unexp_voronoi(a), a.explored_stage, agents)
    elif algo == 'cu_jgr':
      if not voronoi_mode:
        a.goal = cost_utility_jgr(a.x, a.y, unexpl_coords, a.explored_stage, a.view_range, lambda_=lambda_)
      else:
        a.goal = cost_utility_jgr(a.x, a.y, find_unexp_voronoi(a), a.explored_stage, a.view_range, lambda_=lambda_)
    elif algo == 'new_cu_same' or algo == 'new_cu_hedac_same':
      if not voronoi_mode:
        a.goal = cost_utility_new(a.x, a.y, unexpl_coords, a.explored_stage, agents, a.view_range, algo=algo)
      else:
        a.goal = cost_utility_new(a.x, a.y, find_unexp_voronoi(a), a.explored_stage, agents, a.view_range, algo=algo)


def move_nf_coverage(start_grid, start_agents, coverage_finish = 1.0, debug=True, algo='nf', lambda_=0.8, save_images=False):

  grid = copy.deepcopy(start_grid)

  agents = copy.deepcopy(start_agents)

  for agent in agents:
      grid[agent.x, agent.y] = 2  # Mark initial agent positions
  total_explored = update_total_explored(agents, True)

  update_goals(agents, total_explored, True, algo=algo, lambda_=lambda_)  # create new goal.

  avg_eps_time = []
  rounds = 0
  sum_dist = 0

  # tmp_stages = []

  while calculate_expl_percentage(total_explored) < coverage_finish:
      rounds += 1
      eps_start_time = time.time()
      path_none = 0
      for agent in agents:
          path = AStar(agent.explored_stage, coverage_mode=True).search((agent.x, agent.y), agent.goal)
          if debug:
            # print(len(agents), len(np.argwhere(total_explored == -1)))
            draw_maze(agent.explored_stage, path=path, save_gif=save_images)

          if path and len(path) > 1:
              if (agent.x, agent.y) != path[1]:
                sum_dist += 1
                # print("Sum_dist:", sum_dist)
              grid[agent.x, agent.y] = 0  # Mark the old position as unoccupied
              agent.x, agent.y = path[1]  # Update agent position
              grid[agent.x, agent.y] = 2  # Mark the new position as occupied by agent
          else:
            path_none += 1
          agent.agent_view(start_grid)
          total_explored = update_total_explored(agents, True)

      # tmp_stages.append(copy.deepcopy(total_explored))
      # update goals (if agents have explored the goals of another agent):
      update_goals(agents, total_explored, algo=algo, lambda_=lambda_)
      # print("Rounds:", rounds)

      avg_eps_time.append(time.time() - eps_start_time)
      if path_none >= len(agents): # stops if no agents have moved.
          # print(path_none, len(agents))
          print("STOP PATH NONE")
          break

  re = check_real_expl(start_grid, total_explored)  # gets the difference of explored & real grid

  return re[1], rounds, total_explored, np.mean(avg_eps_time), sum_dist, calc_exploration_efficiency(total_explored, sum_dist)

def flood_fill(expl_maze, start):
    # function inputs: expl_maze = explored maze of agent, start = the position of the agent.
    maze = copy.deepcopy(expl_maze)
    maze = np.where(maze == 2, 1, maze)
    maze = np.where(maze == -1, 0, maze)
    distances = np.full_like(maze, fill_value=np.iinfo(np.int32).max, dtype=np.float64)
    distances[start] = 0
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def fill(x, y, distance):
        distances[x, y] = distance
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                if maze[nx, ny] == 0 and distances[nx, ny] > distance + 1:
                    fill(nx, ny, distance + 1)
    fill(start[0], start[1], 0)
    distances[distances == np.iinfo(np.int32).max] = np.inf

    # if no where left to go, stays where it is. Else, goes away from start pos.
    distances[start] = np.inf
    if np.all(distances == np.inf):
        distances[start] = 0

    return distances

def find_ff_maxutil(u, agent, min_indices_list) -> tuple:
    max_u = -np.inf
    next_pos = (agent.x, agent.y)
    for i in min_indices_list:
        if max_u < u[i]:
            next_pos = i
            max_u = u[i]
    return next_pos

def select_flood_fill(algo, agent, all_agents, total_explored, stepped_cells, min_indices_list, alpha=10, max_iter=100):
    if algo == 'ff_default':
        cost = [stepped_cells[min_ind] if min_ind in stepped_cells else 0 for min_ind in min_indices_list]
        min_cost_index = cost.index(min(cost))
        return min_indices_list[min_cost_index]

    if algo == 'new_ff_hedac':
        u = calc_attractive_field(total_explored, alpha, max_iter)
        return find_ff_maxutil(u, agent, min_indices_list)

    if algo == 'new_ff_cu_diffgoal':
        u = calc_new_util(all_agents, total_explored, hedac=False)
    elif algo == 'new_ff_cu_hedac_diffgoal':
        u = calc_new_util(all_agents, total_explored, hedac=True)

    visited_matrix = np.zeros_like(total_explored)
    for c in stepped_cells:
        visited_matrix[c] = stepped_cells[c]
    visited_matrix = norm_values(visited_matrix) # normalizes visited matrix scores to range [0, 1].

    u = u - visited_matrix
    return find_ff_maxutil(u, agent, min_indices_list)

def move_ff_coverage(start_grid, start_agents, algo='default', coverage_finish = 1.0, debug=False, save_images=False):
    if algo not in ['ff_default', 'new_ff_hedac', 'new_ff_cu_diffgoal', 'new_ff_cu_hedac_diffgoal']:
        warnings.warn(f"Requested flood fill algorithm '{algo}' has not been implemented. Implementing 'default' flood fill.")
        algo = 'ff_default'

    grid = copy.deepcopy(start_grid)
    agents = copy.deepcopy(start_agents)

    for agent in agents:
        grid[agent.x, agent.y] = 2  # Mark initial agent positions
    total_explored = update_total_explored(agents, True)
    stepped_cells = {(a.x, a.y): 1 for a in agents}

    avg_eps_time = []
    rounds = 0
    sum_dist = 0

    while calculate_expl_percentage(total_explored) < coverage_finish:
        rounds += 1
        eps_start_time = time.time()
        for a in agents:
            old_agent_pos = (a.x, a.y)
            dist = flood_fill(total_explored, (a.x, a.y))
            # selects the smallest distance:
            min_indices = np.where(dist == np.min(dist))
            min_indices_list = list(zip(min_indices[0], min_indices[1]))
            if len(min_indices_list) == 1:
                a.x, a.y = min_indices_list[0]
                stepped_cells[(a.x, a.y)] = stepped_cells.get((a.x, a.y), 0) + 1
            else:
                # in case of equality, selects the one that has been less stepped on.
                # if candidate cells have been stepped the same, select first option.
                a.x, a.y = select_flood_fill(algo, a, agents, total_explored, stepped_cells, min_indices_list)
                stepped_cells[(a.x, a.y)] = stepped_cells.get((a.x, a.y), 0) + 1

            if debug:
                draw_maze(total_explored, goal=(a.x, a.y), save_gif=save_images)
            a.agent_view(start_grid)
            total_explored = update_total_explored(agents, True)
            if old_agent_pos != (a.x, a.y):
                sum_dist += 1
        avg_eps_time.append(time.time() - eps_start_time)

    re = check_real_expl(start_grid, total_explored)  # gets the difference of explored & real grid

    return re[1], rounds, total_explored, np.mean(avg_eps_time), sum_dist, calc_exploration_efficiency(total_explored, sum_dist)

def voronoi_map(agents):
    """
    Returns a 2d matrix representing the regions of each agent.
    The 2d matrix contains the index (from the agents list) of the agent assigned to it.
    """
    D = np.full_like(agents[0].explored_stage, np.inf, dtype=np.float64)
    # I = copy.deepcopy(D)
    I = np.full_like(D, -1)
    for i in range(1, D.shape[0]-1):
        for j in range(1, D.shape[1]-1):
            x_k = (i, j)
            for a_ind, a in enumerate(agents):
                x_s = (a.x, a.y)
                distance = abs(x_s[0] - x_k[0]) + abs(x_s[1] - x_k[1])
                if distance < D[x_k]:
                    I[x_k] = a_ind
                    D[x_k] = distance
    return I

def broadcast_explored(agents):
    for a in agents:
        a.explored_stage[a.explored_stage == 2] = 0 # removes all known agent positions.
        a.explored_stage[(a.x, a.y)] = 2

    for a_i in agents:
        for a_j in agents:
            distance = abs(a_i.x - a_j.x) + abs(a_i.y - a_j.y)
            if distance > 0 and distance <= a_i.broadcast_range:   # diffrent agents & within broadcast
                a_i.explored_stage[a_i.explored_stage == -1] = a_j.explored_stage[a_i.explored_stage == -1]
                # shares agent positions (if in broadcast range):
                a_i.explored_stage[a_i.x, a_i.y] = 2
                a_i.explored_stage[a_j.x, a_j.y] = 2
                # shares explored stage.
                a_j.explored_stage = copy.deepcopy(a_i.explored_stage)

def find_agent_vcoords(matrix, agent_indx):
    coords = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == agent_indx:
                coords.append((i, j))
    return coords

def update_voronoi_regions(agents, total_explored, v_map):
    v_map_tmp = copy.deepcopy(v_map)

    for a_i, agent in enumerate(agents):
        finish_a = True
        for v in list(agent.voronoi_coords):
            # if total_explored[v] == -1:
            if agent.explored_stage[v] == -1:
                # print(f'Agent {a_i} has not finished its partition. Remaining cells:')
                # print(find_unexp_voronoi(agent))
                finish_a =  False  # agent has not finished exploring its region.
                break

        if not finish_a:    # agent has not finished exploring its region -> does not update region.
            continue

        # print(f'Agent {a_i} has finished its partition ({v_map[(agent.x, agent.y)]})!')

        # v_map_tmp[agent.explored_stage != -1] = -1
        v_map_tmp[total_explored != -1] = -1
        min_distance = np.inf
        min_v_part = -1

        for i in range(v_map_tmp.shape[0]):
            for j in range(v_map_tmp.shape[1]):
                if v_map_tmp[(i, j)] != -1:
                    path = AStar(agent.explored_stage, coverage_mode=True).search((agent.x, agent.y), (i, j))
                    if path is not None and len(path) < min_distance:
                        min_distance = len(path)
                        min_v_part = v_map_tmp[(i, j)]

        # print(f'  Agent {a_i} new partition is {min_v_part}\n    Old Agent {a_i} partition coords: {agent.voronoi_coords}')
        agent.voronoi_coords = find_agent_vcoords(v_map, min_v_part)
        # print(f'    New Agent {a_i} partition coords: {agent.voronoi_coords}')

def find_near_agents(agents) -> list:
    near_agents = []
    for a_i in agents:
        agent_i_list = [a_i]
        for a_j in agents:
            distance = abs(a_i.x - a_j.x) + abs(a_i.y - a_j.y)
            if distance > 0 and distance <= a_i.broadcast_range:
                if len(agent_i_list) <= 1:
                    agent_i_list.append(a_j)
                else:
                    append_list = True
                    for a_k in agent_i_list:
                        if a_j in agent_i_list:
                            continue
                        distance2 = abs(a_k.x - a_j.x) + abs(a_k.y - a_j.y)
                        if distance2 > a_i.broadcast_range:
                            append_list = False
                            break
                    if append_list:
                        agent_i_list.append(a_j)

        agent_i_list.sort(key=lambda x: (x.x, x.y))  # Sort by coordinates (x, y)
        if agent_i_list not in near_agents:
            near_agents.append(agent_i_list)

    return near_agents

def see_goals(agents, expl_perc):
    for a in agents:
      if expl_perc < 1.0 and a.goal != (a.x, a.y) and a.goal not in a.voronoi_coords:
        print("FAILS")
        print(f'Agent at {(a.x, a.y)} has goal {a.goal}. V_coords: {a.voronoi_coords}')
        print("================")
        break

def move_voronoi_coverage(start_grid, start_agents, coverage_finish = 1.0, debug=True, algo='nf', lambda_=0.8, save_images=False):

  grid = copy.deepcopy(start_grid)
  agents = copy.deepcopy(start_agents)
  near_agents = find_near_agents(agents)

  v_map = voronoi_map(agents)
  if debug:
    draw_maze_voronoi(v_map, save_gif=save_images)

  for i, agent in enumerate(agents):
      agent.voronoi_coords = find_agent_vcoords(v_map, i)
      grid[agent.x, agent.y] = 2  # Mark initial agent positions
      agent.explored_stage[agent.explored_stage == 2] = 0
      agent.explored_stage[(agent.x, agent.y)] = 2
      # fills the border with 1s
      agent.explored_stage[0, :] = 1
      agent.explored_stage[-1, :] = 1
      agent.explored_stage[:, 0] = 1
      agent.explored_stage[:, -1] = 1

  broadcast_explored(agents)
  total_explored = update_total_explored(agents, True, True)

  update_voronoi_regions(agents, total_explored, v_map)
  for agents_list in near_agents:
    update_goals(agents_list, total_explored, True, algo=algo, lambda_=lambda_, voronoi_mode=True)  # create new goal.
    see_goals(agents_list, calculate_expl_percentage(total_explored))

  avg_eps_time = []
  rounds = 0
  sum_dist = 0
  while calculate_expl_percentage(total_explored) < coverage_finish:
      rounds += 1
      eps_start_time = time.time()
      path_none = 0
      for i, agent in enumerate(agents):
          path = AStar(agent.explored_stage, coverage_mode=True).search((agent.x, agent.y), agent.goal)
          if debug:
            draw_maze_voronoi(v_map, agent.explored_stage, path=path, save_gif=save_images)

          if path and len(path) > 1:
              if (agent.x, agent.y) != path[1]:
                sum_dist += 1
              grid[agent.x, agent.y] = 0  # Mark the old position as unoccupied
              agent.x, agent.y = path[1]  # Update agent position
              grid[agent.x, agent.y] = 2  # Mark the new position as occupied by agent
          else:
            path_none += 1
          agent.agent_view(start_grid)
          broadcast_explored(agents)
          total_explored = update_total_explored(agents, True, True)
      if debug:
        print("ENTIRE EXPLORED ==========")
        draw_maze(total_explored, save_gif=save_images)
        print("=============")

      # update goals (if agents have explored the goals of another agent):
      update_voronoi_regions(agents, total_explored, v_map)
      near_agents = find_near_agents(agents)
      for agents_list in near_agents:
        # update goals (if agents have explored the goals of another agent):
        update_goals(agents_list, total_explored, True, algo=algo, lambda_=lambda_, voronoi_mode=True)
        see_goals(agents_list, calculate_expl_percentage(total_explored))

      avg_eps_time.append(time.time() - eps_start_time)
      if path_none >= len(agents): # stops if no agents have moved.
          # print(path_none, len(agents))
          print("STOP PATH NONE")
          break

  re = check_real_expl(start_grid, total_explored)  # gets the difference of explored & real grid

  return re[1], rounds, total_explored, np.mean(avg_eps_time), sum_dist, calc_exploration_efficiency(total_explored, sum_dist)


def save_xlsx(file_path: str, new_row: dict):
  """
  Saves new_row to specified xlsx file_path.
  Example of new_row dict:
    new_row = {
      "#_Agents": num_agent,
      "Coverage": avg_cover,
      "Finished_Agents": avg_finish,
      "Experiment_Time": avg_expt_time,
      "Episode_Time": avg_eps_time,
      "Agent_Finish_Time": avg_agent_time,
      "Dimensions": (maze_dim, maze_dim)
  }
  """

  try:
      df = pd.read_excel(file_path)
  except FileNotFoundError:
      df = pd.DataFrame(columns=[k for k in new_row.keys()])
  df = df._append(new_row, ignore_index=True)
  df.to_excel(file_path, float_format='%.5f', index=False)

"""Function to test astar with many experiments (print averages)."""

def test_astar(num_agents, num_test, start_grid = None, gen_stage_func = None, file_path = None, agent_view_range = 2, debug=False, coverage_mode=True, coverage_finish=1.0, algo='nf', alpha=10, max_hedac_iter=100, lambda_=0.8, voronoi_mode=False, save_images=False):
  """
  Function to test astar with many experiments (returns averages).
  If you want to give the initial grid, initialize the variable start_grid with your grid
  (and put the gen_stage_func parameter = None).
  If you want to create a different stage for each experiment, put start_grid = None
  and initialize the next parameter (gen_stage_func) like so:
  functools.partial(func_for_grid_gen, rows, cols, obs_prob)
  where func_for_grid_gen is a function for grid generation, rows and cols are the number
  of rows and columns of the grid and obs_prob the probablity of obstacles.
  """
  if save_images:
    debug = True
  avg_cover = []
  total_rounds = []
  avg_exp_time = []
  avg_round_time= []
  avg_agent_step_time = []
  avg_expl_cost = []
  avg_expl_eff = []

  params = gen_stage_func.keywords

  count_false = 0

  for i in range(num_test):
    if count_false > 4:
      break

    print(f"Test: {i}")
    start_time = time.time()
    if start_grid is None:
      grid = gen_stage_func()
    else:
      grid = copy.deepcopy(start_grid)
    agents = generate_agents(real_stage = grid, num_agents = num_agents, view_range = agent_view_range, coverage_mode = coverage_mode)
    if coverage_mode:
      if algo == 'hedac':
        res = move_hedac_coverage(start_grid=grid, agents=agents, alpha=alpha, coverage_finish=coverage_finish, max_iter=max_hedac_iter, debug=debug, save_images=save_images)
      elif algo in ['ff_default', 'new_ff_hedac', 'new_ff_cu_diffgoal', 'new_ff_cu_hedac_diffgoal']:
        res = move_ff_coverage(start_grid=grid, start_agents=agents, algo=algo, coverage_finish=coverage_finish, debug=debug, save_images=save_images)
      elif not voronoi_mode:
        res = move_nf_coverage(start_grid=grid, start_agents=agents, coverage_finish = coverage_finish, debug=debug, algo=algo, lambda_=lambda_, save_images=save_images)
      else:
        res = move_voronoi_coverage(start_grid=grid, start_agents=agents, coverage_finish= coverage_finish, debug=debug, algo=algo, lambda_=lambda_, save_images=save_images)
      # if res[0] != 1.0:
      #   count_false += 1
      if file_path is not None:
        save_xlsx(file_path, {"#_Agents":num_agents, "Coverage": res[0], "Total_Rounds": res[1], "Expl_Cost": res[4], "Expl_Eff": res[5], "Avg_Round_Time": res[3], "Avg_Agent_Step_Time": res[3]/num_agents, "Experiment_Time": time.time()-start_time, "Obs_Prob": params["obs_prob"], "Test": i})
      avg_expl_cost.append(res[4])
      avg_expl_eff.append(res[5])
    else:
      res = move_astar(start_grid=grid, start_agents=agents, debug=debug)
      if file_path is not None:
        save_xlsx(file_path, {"#_Agents":num_agents, "Coverage": res[0], "Total_Rounds": res[1], "Avg_Rounds": res[2], "Avg_Round_Time": res[3], "Finished_Agents": res[4], "Avg_Agent_Step_Time": res[3]/num_agents, "Experiment_Time": time.time()-start_time, "Obs_Prob": params["obs_prob"], "Test": i})
    avg_cover.append(res[0])
    total_rounds.append(res[1])
    avg_exp_time.append(time.time() - start_time)
    avg_round_time.append(res[3])
    avg_agent_step_time.append(res[3]/num_agents)
    # if res[0] != 1:
    #   draw_maze(res[2])

  avg_cover, avg_rounds, avg_exp_time, avg_round_time, avg_agent_step_time, std_rounds = np.mean(avg_cover), np.mean(total_rounds), np.mean(avg_exp_time), np.mean(avg_round_time), np.mean(avg_agent_step_time), np.std(total_rounds)
  if coverage_mode:
    avg_expl_cost, avg_expl_eff = np.mean(avg_expl_cost), np.mean(avg_expl_eff)
    print(f"Average Coverage Percentage: {avg_cover} / Average Total Rounds: {avg_rounds} / Std Total Rounds: {std_rounds} / Average Expl Cost {avg_expl_cost} / Average Expl Efficiency {avg_expl_eff} / Average Round Time: {avg_round_time} / Average Agent Step Time: {avg_agent_step_time} / Average Experiment Time: {avg_exp_time}")
    return avg_cover, avg_rounds, avg_round_time, avg_agent_step_time, avg_exp_time, std_rounds, avg_expl_cost, avg_expl_eff

  # not coverage mode:
  print(f"Average Coverage Percentage: {avg_cover} / Average Total Rounds: {avg_rounds} / Std Total Rounds: {std_rounds} / Average Round Time: {avg_round_time} / Average Agent Step Time: {avg_agent_step_time} / Average Experiment Time: {avg_exp_time}")
  return avg_cover, avg_rounds, avg_round_time, avg_agent_step_time, avg_exp_time, std_rounds


def run_exp_xlsx(file_path, file_path_all, agents_num_list, rows, cols, num_test, obs_prob=0.85, agent_view = 2, coverage_mode=True, algo='nf', alpha=10, max_hedac_iter=100, lambda_=0.8, voronoi_mode=False):
  """Function to test and save the results of the algos."""
  try:
      df = pd.read_excel(file_path)
  except FileNotFoundError:
    if not coverage_mode:
      df = pd.DataFrame(columns=["#_Agents", "Coverage", "Avg_Total_Rounds", "Avg_Round_Time", "Avg_Agent_Step_Time", "Experiment_Time", "Obs_Prob"])
    else:
      df = pd.DataFrame(columns=["#_Agents", "Coverage", "Avg_Total_Rounds", "Avg_Expl_Cost", "Avg_Expl_Eff", "Avg_Round_Time", "Avg_Agent_Step_Time", "Experiment_Time", "Obs_Prob"])

  # print(f"DEBUG {agents_num_list} ==========")
  for num_agent in agents_num_list:
    print(f"Agent number: {num_agent} / Obs_Prob: {obs_prob}")
    res = test_astar(num_agent, num_test, gen_stage_func=functools.partial(create_maze, rows=rows, cols=cols, obs_prob=obs_prob), agent_view_range=agent_view, debug=False, file_path=file_path_all, coverage_mode=coverage_mode, algo=algo, alpha=alpha, max_hedac_iter=max_hedac_iter, lambda_=lambda_, voronoi_mode=voronoi_mode, save_images=False)
    avg_cover, avg_rounds, avg_round_time, avg_agent_step_time, avg_exp_time, std_rounds = res[0], res[1], res[2], res[3], res[4], res[5]
    new_row = {
        "#_Agents": num_agent,
        "Coverage": avg_cover,
        "Avg_Total_Rounds": avg_rounds,
        "Std_Total_Rounds": std_rounds,
        "Avg_Round_Time": avg_round_time,
        "Avg_Agent_Step_Time": avg_agent_step_time,
        "Experiment_Time": avg_exp_time,
        "Obs_Prob": obs_prob,
    }
    if coverage_mode:
      new_row['Avg_Expl_Cost'] = res[6]
      new_row['Avg_Expl_Eff'] = res[7]
    df = df._append(new_row, ignore_index=True)
    df.to_excel(file_path, float_format='%.5f', index=False)
    print(f"Agent number: {num_agent} / Obs_Prob: {1-obs_prob}")
    res = test_astar(num_agent, num_test, gen_stage_func=functools.partial(create_maze, rows=rows, cols=cols, obs_prob= 1 - obs_prob), agent_view_range=agent_view, debug=False, file_path=file_path_all, coverage_mode=coverage_mode, algo=algo, alpha=alpha, max_hedac_iter=max_hedac_iter, lambda_=lambda_, voronoi_mode=voronoi_mode, save_images=False)
    avg_cover, avg_rounds, avg_round_time, avg_agent_step_time, avg_exp_time, std_rounds = res[0], res[1], res[2], res[3], res[4], res[5]
    new_row = {
        "#_Agents": num_agent,
        "Coverage": avg_cover,
        "Avg_Total_Rounds": avg_rounds,
        "Std_Total_Rounds": std_rounds,
        "Avg_Round_Time": avg_round_time,
        "Avg_Agent_Step_Time": avg_agent_step_time,
        "Experiment_Time": avg_exp_time,
        "Obs_Prob": 1 - obs_prob,
    }
    if coverage_mode:
      new_row['Avg_Expl_Cost'] = res[6]
      new_row['Avg_Expl_Eff'] = res[7]
    df = df._append(new_row, ignore_index=True)
    df.to_excel(file_path, float_format='%.5f', index=False)

def run_all_exp(algo, agents_num_list, rows, cols, num_test, obs_prob=0.85, agent_view=2, coverage_mode=True, alpha=10, max_hedac_iter=100, lambda_=0.8, voronoi_mode=False):
  """Final function to run all experiments."""
  if not coverage_mode: # this is not researched in the thesis.
    algo = 'astar'
  file_name_all = f'{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}.xlsx'
  file_name = f'{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}.xlsx'  # saves the averages.

  parent_parent_dir = "results"
  if voronoi_mode:
    parent_parent_dir = "results_submaps"

  if check_run_colab(): # runs all:
    file_path_all = f'/content/drive/My Drive/thesis/{parent_parent_dir}/{algo}/{rows}x{cols}/all/' + file_name_all
    file_path = f'/content/drive/My Drive/thesis/{parent_parent_dir}/{algo}/{rows}x{cols}/' + file_name  # saves the averages.
  else:
    parent_dir = f"{algo}"
    sub_dir = f"{rows}x{cols}"
    sub_sub_dir = "all"
    os.makedirs(parent_parent_dir, exist_ok=True)
    os.makedirs(os.path.join(parent_parent_dir, parent_dir), exist_ok=True)
    os.makedirs(os.path.join(parent_parent_dir, parent_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(parent_parent_dir, parent_dir, sub_dir, sub_sub_dir), exist_ok=True)
    file_path = os.path.join(parent_parent_dir, parent_dir, sub_dir, file_name)
    file_path_all = os.path.join(parent_parent_dir, parent_dir, sub_dir, sub_sub_dir, file_name_all)

  print(f"Running {algo} algorithm...")
  if get_ipython() and os.name != 'posix': # runs in jupiter and windows
    run_exp_xlsx(file_path, file_path_all, agents_num_list, rows, cols, num_test, obs_prob, agent_view, coverage_mode, algo, lambda_=lambda_, voronoi_mode=voronoi_mode)
  else: # in unix or script (not notebook) -> uses multiprocesses:
    # num_cores = 4
    num_cores = multiprocessing.cpu_count()
    if num_cores >= 8:
      num_cores = num_cores // 2 # min num cores is 4 (good number for number of agents)
    file_paths = [os.path.join(parent_parent_dir, parent_dir, sub_dir, f"p{i+1}_" + file_name) for i in range(num_cores)]
    file_paths_all = [os.path.join(parent_parent_dir, parent_dir, sub_dir, sub_sub_dir, f"p{i+1}_" + file_name_all) for i in range(num_cores)]

    # Calculating the agent distribution in the processes:
    team_size = 1 + max(0, len(agents_num_list) - num_cores)
    agent_teams = [agents_num_list[i:i+team_size] for i,_ in enumerate(agents_num_list)]
    while len(agent_teams) > num_cores:
      agent_teams.pop()
    l = len(agent_teams)
    while len(agent_teams) < num_cores:
      i = random.randint(0, l - 1)
      agent_teams.append([agents_num_list[i]])
    print(f"Agent Teams: {agent_teams}")
    # ------------------------

    l = len(agent_teams[0]) # gets the team size of each team
    params_list = [(file_paths[i], file_paths_all[i], agent_teams[i], rows, cols, num_test, round(abs((i % 2) - obs_prob), 2), agent_view, coverage_mode, algo, alpha, max_hedac_iter, lambda_, voronoi_mode) for i in range(num_cores)]
    print(params_list)

    if __name__ == "__main__":
      processes = []
      for params in params_list:
          process = multiprocessing.Process(target=run_exp_xlsx, args=params)
          processes.append(process)

      for process in processes:
          process.start()

      for process in processes:
          process.join()

      if os.name == 'posix' or not get_ipython():  # combines unix files to one:
        df = []
        df_all = []
        for i in range(1, num_cores+1):
            # path_all = f'{parent_parent_dir}/{algo}/{rows}x{cols}/all/p{i}_{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}.xlsx'
            path_all = os.path.join(parent_parent_dir, algo, f"{rows}x{cols}", "all", f"p{i}_{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}.xlsx")
            # path_avg = f'{parent_parent_dir}/{algo}/{rows}x{cols}/p{i}_{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}.xlsx'
            path_avg = os.path.join(parent_parent_dir, algo, f"{rows}x{cols}", f"p{i}_{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}.xlsx")
            df_all.append(pd.read_excel(path_all))
            df.append(pd.read_excel(path_avg))
            os.remove(path_avg)
            os.remove(path_all)

        # path_all = f'{parent_parent_dir}/{algo}/{rows}x{cols}/all/p_{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}'
        path_all = os.path.join(parent_parent_dir, algo, f"{rows}x{cols}", "all", f"p_{algo}_all_{rows}x{cols}_{coverage_mode}_{agent_view}")
        # path = f'{parent_parent_dir}/{algo}/{rows}x{cols}/p_{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}'
        path = os.path.join(parent_parent_dir, algo, f"{rows}x{cols}", f"p_{algo}_{rows}x{cols}_{coverage_mode}_{agent_view}_{num_test}")

        if algo == 'cu_jgr':
           path_all += f"_{lambda_}"
           path += f"_{lambda_}"

        df_all = pd.concat(df_all, ignore_index=True, axis=0)
        if os.path.exists(path_all+".xlsx"):
          df_tmp = pd.read_excel(path_all+".xlsx")
          df_all = pd.concat([df_tmp, df_all], axis=0)
        df_all.to_excel(path_all+".xlsx", index=False)

        df = pd.concat(df, ignore_index=True, axis=0)
        if os.path.exists(path+".xlsx"):
          df_tmp = pd.read_excel(path+".xlsx")
          df = pd.concat([df_tmp, df], axis=0)
        df.to_excel(path+".xlsx", index=False)

"""Running the experiments:"""

# Initialization Parameters ========
agents_num_list = [[1, 2, 4, 6], [8, 10]]
rows = 15
cols = 15
num_test = 450
obs_prob = 0.85
agent_view = 2
coverage_mode = True    # 'coverage_mode = True' is researched in the thesis.
alpha, max_hedac_iter = 10, 100 # used in hedac
lambda_ = 0.8 # used in cost-utility jgr
voronoi_mode = False
algos = ['new_ff_cu_hedac_diffgoal']
for t_algo in algos:
  for agents_num_list_i in agents_num_list:
    run_all_exp(t_algo, agents_num_list_i, rows, cols, num_test, obs_prob, agent_view, coverage_mode, alpha, max_hedac_iter, lambda_, voronoi_mode=voronoi_mode)
