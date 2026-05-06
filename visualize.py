import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

INPUT_DIR = "input"
LOG_FILE = "results_clu7.log"
FIGS_DIR = "figs"

os.makedirs(FIGS_DIR, exist_ok=True)


def parse_vrp(filepath):
    """Return list of (demand, x, y) for each node (index 0 = depot)."""
    nodes = []
    with open(filepath) as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            nodes.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return nodes


def parse_routes(solution_str):
    """Split solution string into list of routes (each route is a list of customer ids)."""
    tokens = list(map(int, solution_str.split()))
    routes = []
    current = []
    # solution starts and ends with 0; routes are separated by "0 0"
    i = 0
    while i < len(tokens):
        if tokens[i] == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(tokens[i])
        i += 1
    if current:
        routes.append(current)
    return routes


def plot_solution(instance_name, nodes, routes, cost, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    depot_x, depot_y = nodes[0][1], nodes[0][2]
    cust_x = [n[1] for n in nodes[1:]]
    cust_y = [n[2] for n in nodes[1:]]

    colors = cm.tab20(np.linspace(0, 1, max(len(routes), 1)))

    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        xs = [depot_x] + [nodes[c][1] for c in route] + [depot_x]
        ys = [depot_y] + [nodes[c][2] for c in route] + [depot_y]
        ax.plot(xs, ys, color=color, linewidth=1.2, zorder=1)

    ax.scatter(cust_x, cust_y, color="steelblue", s=30, zorder=2)
    ax.scatter([depot_x], [depot_y], color="red", s=120, marker="*", zorder=3, label="Depot")

    ax.set_title(f"{instance_name}  |  cost={cost:.1f}", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


with open(LOG_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        instance = entry["Instance"]
        cost = entry["Result"]
        solution_str = entry["Solution"]

        vrp_path = os.path.join(INPUT_DIR, instance)
        if not os.path.exists(vrp_path):
            print(f"WARNING: {vrp_path} not found, skipping.")
            continue

        nodes = parse_vrp(vrp_path)
        routes = parse_routes(solution_str)

        name_stem = os.path.splitext(instance)[0]
        out_path = os.path.join(FIGS_DIR, f"{name_stem}.png")
        plot_solution(name_stem, nodes, routes, cost, out_path)
