import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from collections import deque

# 定义邻接矩阵
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

# 构造图
G = nx.from_numpy_array(A)

# BFS 函数，记录遍历顺序
def bfs_steps(G, start=0):
    visited = [False] * len(G)
    queue = deque([start])
    visited[start] = True
    steps = []

    while queue:
        node = queue.popleft()
        steps.append(("visit", node))
        for neighbor in G.neighbors(node):
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                steps.append(("discover", (node, neighbor)))
    return steps

steps = bfs_steps(G, start=0)

# 可视化设置
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots()
node_colors = ["lightgray"] * len(G)
edge_colors = ["lightgray"] * len(G.edges())

def update(frame):
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, ax=ax)

    action, item = steps[frame]
    if action == "visit":
        node_colors[item] = "lightblue"
    elif action == "discover":
        u, v = item
        edge_index = list(G.edges()).index((u, v) if (u, v) in G.edges() else (v, u))
        edge_colors[edge_index] = "red"

ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat=False)
plt.show()
