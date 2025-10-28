# =========================
# Tabu Search — TSP Manhattan (no diagonals)
# Instance: 1=(3,8) 2=(7,0) 3=(0,2) 4=(0,5) 5=(8,4)
# Tour initial: [1, 2, 3, 4, 5] (retour au début implicite)
# =========================
from math import inf

# ---- Données
coords = {
    1: (3, 8),
    2: (7, 0),
    3: (0, 2),
    4: (0, 5),
    5: (8, 4),
}

def manhattan(a, b):
    xa, ya = coords[a]
    xb, yb = coords[b]
    return abs(xa - xb) + abs(ya - yb)

# ---- Utilitaires TSP
def tour_cost(tour):
    """tour: liste sans retour (ex: [1,2,3,4,5]); on boucle au début."""
    n = len(tour)
    c = 0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        c += manhattan(u, v)
    return c

def edge_key(u, v):
    """Arête non orientée pour la liste tabou."""
    return tuple(sorted((u, v)))

def edges_of_tour(tour):
    n = len(tour)
    E = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        E.append(edge_key(u, v))
    return E

# ---- 2-opt
def two_opt_apply(tour, i, j):
    """
    Applique 2-opt entre indices i..j (i < j), renvoie (new_tour, removed_edges).
    On suppose tour sans retour. Le départ (index 0) reste fixe.
    """
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    removed = [edge_key(a, b), edge_key(c, d)]
    new = tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
    return new, removed

def generate_two_opt_moves(tour):
    """
    Génère tous les couples (i, j) valides pour 2-opt
    en gardant le premier sommet fixe à la position 0.
    """
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n - (0 if i > 1 else 1)):
            # Évite de couper des arêtes adjacentes triviales et la "grande" arête (0,n-1) simultanément
            if j == i:
                continue
            if j == i + 1:
                continue
            # Évite le move qui inverserait tout en cassant le start (0,n-1) + (0,1)
            if i == 1 and j == n - 1:
                continue
            yield (i, j)

# ---- Tabu Search
def tabu_search_tsp(
    tour_init,
    tenure=5,
    max_iter=200,
    verbose=True
):
    """
    Tabu Search avec voisinage 2-opt, liste tabou d'arêtes supprimées,
    aspiration si meilleur global.
    """
    current = tour_init[:]
    best = current[:]
    best_cost = tour_cost(best)

    # Dictionnaire: edge -> iteration d'expiration
    tabu = {}

    if verbose:
        print(f"[Init] tour={current + [current[0]]}  cost={best_cost}")

    for it in range(1, max_iter + 1):
        best_cand = None
        best_cand_cost = inf
        best_cand_removed = None

        # Parcourt du voisinage 2-opt
        for (i, j) in generate_two_opt_moves(current):
            cand, removed = two_opt_apply(current, i, j)
            c = tour_cost(cand)

            # Move admissible ?
            # - Non tabou si toutes les arêtes supprimées sont expirées
            # - Ou bien aspiration si on améliore le meilleur global
            is_tabu = any((e in tabu and tabu[e] > it) for e in removed)
            if is_tabu and c >= best_cost:
                continue  # move interdit

            if c < best_cand_cost:
                best_cand = cand
                best_cand_cost = c
                best_cand_removed = removed

        # Aucun voisin admissible (devrait être rare sur petit cas) : on s'arrête
        if best_cand is None:
            if verbose:
                print(f"[Iter {it}] Aucun voisin admissible — arrêt.")
            break

        # Mettre à jour la solution courante
        current = best_cand

        # Mettre à jour la liste tabou (arêtes supprimées)
        for e in best_cand_removed:
            tabu[e] = it + tenure

        # Purge optionnelle des arêtes expirées (pas obligatoire)
        # tabu = {e: t for e, t in tabu.items() if t > it}

        # Mettre à jour le meilleur global
        if best_cand_cost < best_cost:
            best = current[:]
            best_cost = best_cand_cost
            if verbose:
                print(f"[Iter {it}] NEW BEST  cost={best_cost}  tour={best + [best[0]]}")
        else:
            if verbose and it % 10 == 0:
                print(f"[Iter {it}] cur_cost={best_cand_cost}  best={best_cost}")

    return best, best_cost

# ---- Lancement sur l'instance
if __name__ == "__main__":
    tour0 = [1, 2, 3, 4, 5]  # retour implicite à 1
    best_tour, best_cost = tabu_search_tsp(tour0, tenure=4, max_iter=100, verbose=True)

    print("\n=== Résultat final ===")
    print("Meilleure tournée:", best_tour + [best_tour[0]])
    print("Coût Manhattan   :", best_cost)
