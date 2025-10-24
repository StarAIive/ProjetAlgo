# ============================================================
# ACO (Colonies de Fourmis) pour TSP — Manhattan, pas de diagonales
# Instance : 1=(3,8), 2=(7,0), 3=(0,2), 4=(0,5), 5=(8,4)
# Sorties :
#   - Graphique de l'évolution des coûts
#   - Tracé de la meilleure tournée
#   - Heatmap de la matrice de phéromones
#   - Log CSV (iteration, best_so_far, iter_best, iter_mean, iter_std)
# ============================================================

import math
import random
import csv
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Données (Manhattan)
# -------------------------
coords: Dict[int, Tuple[int, int]] = {
    1: (3, 8),
    2: (7, 0),
    3: (0, 2),
    4: (0, 5),
    5: (8, 4),
}
cities = sorted(coords.keys())
n = len(cities)

def manhattan(a: int, b: int) -> int:
    xa, ya = coords[a]
    xb, yb = coords[b]
    return abs(xa - xb) + abs(ya - yb)

def cycle_cost(tour: List[int]) -> int:
    """tour sans le retour (ex: [1,2,3,4,5]) — on boucle automatiquement."""
    total = 0
    for i in range(len(tour)):
        total += manhattan(tour[i], tour[(i+1) % len(tour)])
    return total

# -------------------------
# Matrices distance / visibilité
# -------------------------
dist = np.zeros((n, n), dtype=float)
for i, ci in enumerate(cities):
    for j, cj in enumerate(cities):
        dist[i, j] = 0.0 if ci == cj else float(manhattan(ci, cj))

visibility = np.zeros_like(dist)
for i in range(n):
    for j in range(n):
        visibility[i, j] = 0.0 if i == j else 1.0 / dist[i, j]

# -------------------------
# Paramètres ACO
# -------------------------
alpha = 1.0     # influence phéromone
beta  = 2.0     # influence visibilité (1/d)
rho   = 0.5     # évaporation
Q     = 100.0   # dépôt de phéromone (normalisé par la longueur)
num_ants  = 20  # fourmis par itération
num_iters = 40  # itérations
start_city = 1  # point de départ (fixé pour reproductibilité)
seed = 7        # seed aléatoire (None pour aléatoire à chaque run)

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)

# Phéromones initiales (symétrique)
tau = np.ones((n, n), dtype=float)

def choose_next(current: int, unvisited: List[int]) -> int:
    """Roulette wheel : P(i->j) ∝ (tau^alpha)*(eta^beta)."""
    i = cities.index(current)
    weights = []
    for city in unvisited:
        j = cities.index(city)
        w = (tau[i, j] ** alpha) * (visibility[i, j] ** beta)
        weights.append(w)
    s = sum(weights)
    probs = [1.0/len(unvisited)] * len(unvisited) if s <= 0 else [w/s for w in weights]
    r, cum = random.random(), 0.0
    for city, p in zip(unvisited, probs):
        cum += p
        if r <= cum:
            return city
    return unvisited[-1]

# -------------------------
# ACO — boucle principale
# -------------------------
best_route: List[int] = None
best_cost = float("inf")

iter_log_rows = []  # pour CSV

for it in range(1, num_iters + 1):
    routes, lengths = [], []

    # Construction des tournées (toutes les fourmis)
    for _ in range(num_ants):
        route = [start_city]
        unvisited = [c for c in cities if c != start_city]
        while unvisited:
            nxt = choose_next(route[-1], unvisited)
            route.append(nxt)
            unvisited.remove(nxt)
        L = cycle_cost(route)
        routes.append(route)
        lengths.append(L)
        if L < best_cost:
            best_cost = L
            best_route = route[:]

    # Évaporation
    tau *= (1.0 - rho)

    # Dépôt (toutes les fourmis)
    for route, L in zip(routes, lengths):
        deposit = Q / L
        for k in range(len(route)):
            a = route[k]
            b = route[(k+1) % len(route)]
            ia, ib = cities.index(a), cities.index(b)
            tau[ia, ib] += deposit
            tau[ib, ia] += deposit  # symétrique

    # Log d'itération
    iter_best = float(np.min(lengths))
    iter_mean = float(np.mean(lengths))
    iter_std  = float(np.std(lengths))
    iter_log_rows.append([it, best_cost, iter_best, iter_mean, iter_std])

# -------------------------
# Sauvegarde CSV
# -------------------------
csv_path = "aco_log.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "best_so_far", "iteration_best", "iteration_mean", "iteration_std"])
    writer.writerows(iter_log_rows)
print(f"[OK] Log CSV écrit : {csv_path}")

# -------------------------
# Graphiques
# -------------------------
its     = [row[0] for row in iter_log_rows]
best_sf = [row[1] for row in iter_log_rows]
iter_b  = [row[2] for row in iter_log_rows]
iter_m  = [row[3] for row in iter_log_rows]

# 1) Évolution des coûts
plt.figure()
plt.plot(its, best_sf, label="Meilleur cumulé")
plt.plot(its, iter_b,  label="Meilleur de l'itération")
plt.plot(its, iter_m,  label="Moyenne de l'itération")
plt.xlabel("Itération")
plt.ylabel("Longueur (Manhattan)")
plt.title("ACO — Évolution des coûts")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Tracé de la meilleure tournée
route_plot = best_route + [best_route[0]]
xs = [coords[c][0] for c in route_plot]
ys = [coords[c][1] for c in route_plot]
plt.figure()
plt.plot(xs, ys, marker="o")
for c in cities:
    x, y = coords[c]
    plt.text(x, y, str(c), ha="right", va="bottom")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"ACO — Meilleure tournée: {route_plot} (L={int(best_cost)})")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Heatmap des phéromones finales
plt.figure()
plt.imshow(tau, interpolation="nearest")
plt.colorbar(label="Phéromone (tau)")
plt.xticks(range(n), cities)
plt.yticks(range(n), cities)
plt.title("ACO — Matrice de phéromones finale")
plt.tight_layout()
plt.show()

print("=== Résultat final ACO ===")
print("Meilleure tournée :", route_plot)
print("Coût (Manhattan)  :", int(best_cost))
