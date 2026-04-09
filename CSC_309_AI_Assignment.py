# ==========================
# Drone Delivery Navigation
# ==========================

import heapq
from collections import deque

# --------- Graph Definition ---------
graph = {
    'A': {'B': 2, 'C': 5, 'D': 1},
    'B': {'A': 2, 'D': 2, 'E': 3},
    'C': {'A': 5, 'D': 2, 'F': 3},
    'D': {'A': 1, 'B': 2, 'C': 2, 'E': 1, 'F': 4},
    'E': {'B': 3, 'D': 1, 'G': 2},
    'F': {'C': 3, 'D': 4, 'G': 1},
    'G': {'E': 2, 'F': 1, 'H': 3},
    'H': {'G': 3}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 6,
    'D': 4,
    'E': 2,
    'F': 2,
    'G': 1,
    'H': 0
}

start = 'A'
goal = 'H'

# ===================================
# Depth-First Search (DFS)
# ===================================
def dfs(graph, start, goal):
    visited = set()
    path = []
    nodes_expanded = [0]

    def dfs_helper(node):
        visited.add(node)
        path.append(node)
        nodes_expanded[0] += 1

        if node == goal:
            return True

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs_helper(neighbor):
                    return True

        path.pop()  # Backtrack
        return False

    dfs_helper(start)
    return path, len(path) - 1, nodes_expanded[0]  # path, steps, nodes expanded


# ===================================
# Breadth-First Search (BFS)
# ===================================
def bfs(graph, start, goal):
    visited = set()
    visited.add(start)
    queue = deque()
    queue.append((start, [start]))
    nodes_expanded = 0

    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1

        if node == goal:
            return path, len(path) - 1, nodes_expanded

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, 0, nodes_expanded  # No path found


# ===================================
# Uniform Cost Search (UCS)
# ===================================
def ucs(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbor, weight in graph[node].items():
            if neighbor not in explored:
                heapq.heappush(frontier, (cost + weight, neighbor, path + [neighbor]))

    return None, 0, nodes_expanded  # No path found


# ===================================
# A* Search
# ===================================
def a_star(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic[start], 0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        f, g, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, g, nodes_expanded

        for neighbor, weight in graph[node].items():
            if neighbor not in explored:
                g_new = g + weight
                f_new = g_new + heuristic[neighbor]
                heapq.heappush(frontier, (f_new, g_new, neighbor, path + [neighbor]))

    return None, 0, nodes_expanded  # No path found


# ===================================
# Optional: Battery-Constrained Search
# ===================================
def ucs_with_battery(graph, start, goal, max_battery):
    """UCS that only explores paths within the battery (cost) limit."""
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if neighbor not in explored and new_cost <= max_battery:
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

    return None, None, nodes_expanded  # No path within battery limit


# ===================================
# Run and Compare
# ===================================
if __name__ == "__main__":

    print("=" * 55)
    print("       DRONE DELIVERY NAVIGATION - A to H")
    print("=" * 55)

    # DFS
    dfs_path, dfs_steps, dfs_expanded = dfs(graph, start, goal)
    print(f"\n🔵 DFS")
    print(f"   Path          : {' → '.join(dfs_path)}")
    print(f"   Steps         : {dfs_steps}")
    print(f"   Nodes Expanded: {dfs_expanded}")

    # BFS
    bfs_path, bfs_steps, bfs_expanded = bfs(graph, start, goal)
    print(f"\n🟢 BFS")
    print(f"   Path          : {' → '.join(bfs_path)}")
    print(f"   Steps         : {bfs_steps}")
    print(f"   Nodes Expanded: {bfs_expanded}")

    # UCS
    ucs_path, ucs_cost, ucs_expanded = ucs(graph, start, goal)
    print(f"\n🟡 UCS")
    print(f"   Path          : {' → '.join(ucs_path)}")
    print(f"   Total Cost    : {ucs_cost}")
    print(f"   Nodes Expanded: {ucs_expanded}")

    # A*
    astar_path, astar_cost, astar_expanded = a_star(graph, start, goal, heuristic)
    print(f"\n🔴 A* Search")
    print(f"   Path          : {' → '.join(astar_path)}")
    print(f"   Total Cost    : {astar_cost}")
    print(f"   Nodes Expanded: {astar_expanded}")

    # Summary Table
    print("\n" + "=" * 55)
    print(f"{'Algorithm':<12} {'Path':<25} {'Cost/Steps':<12} {'Nodes Exp.'}")
    print("-" * 55)
    print(f"{'DFS':<12} {'→'.join(dfs_path):<25} {dfs_steps:<12} {dfs_expanded}")
    print(f"{'BFS':<12} {'→'.join(bfs_path):<25} {bfs_steps:<12} {bfs_expanded}")
    print(f"{'UCS':<12} {'→'.join(ucs_path):<25} {ucs_cost:<12} {ucs_expanded}")
    print(f"{'A*':<12} {'→'.join(astar_path):<25} {astar_cost:<12} {astar_expanded}")
    print("=" * 55)

    # Optional: Battery Constraint Demo
    print("\n⚡ OPTIONAL: Battery-Constrained UCS (max cost = 7)")
    for battery in [7, 8, 10]:
        b_path, b_cost, b_expanded = ucs_with_battery(graph, start, goal, battery)
        if b_path:
            print(f"   Battery={battery}: Path={' → '.join(b_path)}, Cost={b_cost}, Nodes Expanded={b_expanded}")
        else:
            print(f"   Battery={battery}: ❌ No path found within battery limit")
