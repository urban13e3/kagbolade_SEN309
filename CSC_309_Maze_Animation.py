import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq
from collections import deque

# -------------------------------
# Maze Definition
# -------------------------------
#Example 1.
# maze = [
#     ["S", 0, 1, 0, 0, 0, 1, 0, 0, 0],
#     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 1, 1, 1, 0, "G", 0]
# ]

#Example 2.
maze = [
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
[1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
[1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,"G",0,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,"S",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

#Example 3.
# maze = [
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
# [1,"S",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
# [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
# [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
# [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
# [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,"G",0,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
# [1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
# [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# ]

# #Example 4.
# maze = [[1,1,0,0,0,0,1],
#         [1,1,0,1,1,0,1],
#         [1,"G",0,1,0,0,1],
#         [1,0,1,1,0,1,1],
#         [0,0,0,0,0,1,1],
#         ["S",1,1,1,1,1,1]]

#Example 5.
# maze = [
# [0,1,0,1,0,1,1,1,0,0,1,"G"],
# [0,1,0,1,0,0,0,1,0,1,1,0],
# [0,1,0,1,0,1,0,0,0,1,1,0],
# [0,0,0,1,0,1,0,1,0,1,1,0],
# [1,0,0,0,0,1,0,1,0,0,0,0],
# [1,1,1,0,1,1,0,1,1,1,1,1],
# ["S",0,0,0,1,1,0,0,0,0,0,0]
# ]

ROWS = len(maze)
COLS = len(maze[0])

grid = np.zeros((ROWS, COLS))
for i in range(ROWS):
    for j in range(COLS):
        if maze[i][j] == 1:
            grid[i][j] = 1
        elif maze[i][j] == "S":
            start = (i, j)
        elif maze[i][j] == "G":
            goal = (i, j)

# -------------------------------
# Node Class
# -------------------------------
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost  # depth (step cost)

    # Needed so heapq can compare Nodes when costs are equal
    def __lt__(self, other):
        return self.cost < other.cost

# -------------------------------
# Problem Definition
# -------------------------------
def actions(state):
    x, y = state
    moves = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    possible = []
    for action, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            if maze[nx][ny] != 1:
                possible.append(action)
    return possible


def result(state, action):
    moves = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    dx, dy = moves[action]
    return (state[0] + dx, state[1] + dy)


def goal_test(state):
    return state == goal


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path


# -------------------------------
# Heuristic: Manhattan Distance
# -------------------------------
def manhattan(state):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


# -------------------------------
# DFS
# -------------------------------
def dfs_steps():
    frontier = [Node(start, cost=0)]
    explored = set()

    while True:
        if not frontier:
            return

        node = frontier.pop()

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored and \
               all(child_state != n.state for n in frontier):
                frontier.append(Node(child_state, node, action, node.cost + 1))


# -------------------------------
# BFS
# -------------------------------
def bfs_steps():
    frontier = deque([Node(start, cost=0)])
    explored = set()

    while True:
        if not frontier:
            return

        node = frontier.popleft()

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored and \
               all(child_state != n.state for n in frontier):
                frontier.append(Node(child_state, node, action, node.cost + 1))


# -------------------------------
# Greedy Best-First Search
# -------------------------------
def greedy_steps():
    """
    Greedy Best-First Search: expands the node closest to the goal
    using only the heuristic h(n) = Manhattan distance.
    Fast but not guaranteed to find the optimal path.
    """
    counter = 0  # tiebreaker for heapq
    start_node = Node(start, cost=0)
    frontier = [(manhattan(start), counter, start_node)]
    explored = set()
    frontier_states = {start}

    while True:
        if not frontier:
            return

        _, _, node = heapq.heappop(frontier)
        frontier_states.discard(node.state)

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored and child_state not in frontier_states:
                counter += 1
                child_node = Node(child_state, node, action, node.cost + 1)
                heapq.heappush(frontier, (manhattan(child_state), counter, child_node))
                frontier_states.add(child_state)


# -------------------------------
# A* Search
# -------------------------------
def astar_steps():
    """
    A* Search: expands the node with the lowest f(n) = g(n) + h(n),
    where g(n) is the path cost so far and h(n) is the Manhattan distance.
    Guaranteed to find the optimal (shortest) path.
    """
    counter = 0  # tiebreaker for heapq
    start_node = Node(start, cost=0)
    frontier = [(manhattan(start), counter, start_node)]
    explored = set()
    frontier_states = {start: 0}  # state -> best g cost seen

    while True:
        if not frontier:
            return

        _, _, node = heapq.heappop(frontier)

        if node.state in explored:
            continue

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)
            g_new = node.cost + 1
            if child_state not in explored:
                if child_state not in frontier_states or g_new < frontier_states[child_state]:
                    counter += 1
                    child_node = Node(child_state, node, action, g_new)
                    f = g_new + manhattan(child_state)
                    heapq.heappush(frontier, (f, counter, child_node))
                    frontier_states[child_state] = g_new


# -------------------------------
# Animation
# -------------------------------
def animate_solver(algorithm="BFS"):
    fig, ax = plt.subplots()
    ax.set_title(f"AI Maze Solver — {algorithm}", fontsize=12, fontweight="bold")
    ax.set_facecolor("lightgray")

    maze_img = np.copy(grid)
    img = ax.imshow(maze_img, cmap="gray_r", vmin=0, vmax=1)

    ax.scatter(start[1], start[0], c="green", s=100, zorder=5, label="Start")
    ax.scatter(goal[1], goal[0], c="red", s=100, zorder=5, label="Goal")
    ax.legend(loc="upper right", fontsize=8)

    # Text overlays
    frontier_text = ax.text(0, -1, "", fontsize=10)
    explored_text = ax.text(3, -1, "", fontsize=10)

    # Select the correct generator
    algorithm_map = {
        "BFS":    bfs_steps,
        "DFS":    dfs_steps,
        "GREEDY": greedy_steps,
        "ASTAR":  astar_steps,
    }
    steps = algorithm_map.get(algorithm.upper(), bfs_steps)()

    def update(frame):
        nonlocal maze_img

        if isinstance(frame[1], str):
            # Final path highlight
            for x, y in frame[0]:
                maze_img[x][y] = 0.9
        else:
            state, explored_size, frontier_size, cost = frame
            x, y = state
            maze_img[x][y] = 0.5
            frontier_text.set_text(f"Frontier: {frontier_size}")
            explored_text.set_text(f"Explored: {explored_size}")

        img.set_data(maze_img)
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        repeat=False
    )

    plt.tight_layout()
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    # Options: "BFS", "DFS", "GREEDY", "ASTAR"
    animate_solver("ASTAR")
