import numpy as np
class Agent:

  def __init__(self, start: tuple, goal: tuple, real_stage):
    self.x = start[0]
    self.y = start[1]
    self.goal = goal
    self.real_stage = real_stage
    self.explored_stage = np.full_like(real_stage, -1)
    self.explored_stage[self.x, self.y] = 0
    self.agent_view()

  def agent_view(self):
    if self.x > 0:
      self.explored_stage[(self.x - 1, self.y)] = self.real_stage[(self.x - 1, self.y)]
    if self.x < len(self.real_stage) - 1:
      self.explored_stage[(self.x + 1, self.y)] = self.real_stage[(self.x + 1, self.y)]
    if self.y > 0:
      self.explored_stage[(self.x, self.y - 1)] = self.real_stage[(self.x, self.y - 1)]
    if self.y < len(self.real_stage[0]) - 1:
      self.explored_stage[(self.x, self.y + 1)] = self.real_stage[(self.x, self.y + 1)]
    self.explored_stage[self.explored_stage == 2] = 0
