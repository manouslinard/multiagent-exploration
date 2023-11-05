class Node:
    def __init__(self, x, y, g_cost, h_cost):
        self.x = x
        self.y = y
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = None

    def f_cost(self):
        return self.g_cost + self.h_cost

def is_valid(x, y, grid):
    # Checks if the x and y are within grid + if no obs in (x, y).
    rows, cols = grid.shape
    return 0 <= x < rows and 0 <= y < cols and grid[x, y] <= 0

def a_star(grid, start, end):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    rows, cols = grid.shape

    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1], 0, 0)
    open_set.append(start_node)

    while open_set:
        # Finds the lowest f_cost node -> adds it to the closed_set.
        current_node = min(open_set, key=lambda node: node.f_cost())
        open_set.remove(current_node)
        closed_set.add((current_node.x, current_node.y))

        # If current node is the target node -> returns the path.
        if (current_node.x, current_node.y) == end:
            path = []
            while current_node:
                path.insert(0, (current_node.x, current_node.y))
                current_node = current_node.parent
            return path


        for dx, dy in directions:

            # Gets the new pos:
            new_x, new_y = current_node.x + dx, current_node.y + dy

            if is_valid(new_x, new_y, grid) and (new_x, new_y) not in closed_set:
                g_cost = current_node.g_cost + 1  # g_cost (distance) from current node is 1.
                h_cost = abs(new_x - end[0]) + abs(new_y - end[1])
                new_node = Node(new_x, new_y, g_cost, h_cost)
                new_node.parent = current_node

                if new_node not in open_set:
                    open_set.append(new_node)

    return None  # No path found
