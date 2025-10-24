import matplotlib.pyplot as plt

# =========================
# Données (Manhattan)
# =========================
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

def cycle_cost(tour):
    # tour : liste sans le dernier retour (ex: [1,2,3,4,5])
    n = len(tour)
    total = 0
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        total += manhattan(a, b)
    return total

# =========================
# 2-opt : paires valides et opérations
# =========================
def valid_two_opt_pairs(n):
    # Indices i<j avec j - i > 1 et (i=0, j=n-1) exclu (arête de retour)
    for i in range(n - 2):
        for j in range(i + 2, n - 1):
            yield i, j

def two_opt_apply(tour, i, j):
    # Inverse le segment (i+1 .. j) inclus
    new_tour = tour[:]
    new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
    return new_tour

def two_opt_delta(tour, i, j):
    """Retourne (delta_global_local, (A,B,C,D), cost_before_local, cost_after_local)."""
    n = len(tour)
    A = tour[i]
    B = tour[(i+1) % n]
    C = tour[j]
    D = tour[(j+1) % n]
    before = manhattan(A, B) + manhattan(C, D)      # arêtes supprimées
    after  = manhattan(A, C) + manhattan(B, D)      # arêtes ajoutées
    return (after - before), (A, B, C, D), before, after

# =========================
# Visualisation
# =========================
def edge_set(tour):
    n = len(tour)
    s = set()
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        s.add(tuple(sorted((a, b))))
    return s

def plot_cycle(tour, title, highlight_diff=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cyc = tour + [tour[0]]
    xs = [coords[i][0] for i in cyc]
    ys = [coords[i][1] for i in cyc]
    ax.plot(xs, ys, linewidth=2.0, marker="o", alpha=0.9, zorder=1)

    for k, (x, y) in coords.items():
        ax.text(x, y, str(k), fontsize=10, ha="right", va="bottom")

    if highlight_diff is not None:
        added, removed = highlight_diff
        # retirées (pointillé)
        for (u, v) in removed:
            x1, y1 = coords[u]; x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], linestyle="--", linewidth=2.2, zorder=3)
        # ajoutées (plein)
        for (u, v) in added:
            x1, y1 = coords[u]; x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], linewidth=2.8, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("x (cases)")
    ax.set_ylabel("y (cases)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    pad = 1
    xs_all = [p[0] for p in coords.values()]
    ys_all = [p[1] for p in coords.values()]
    ax.set_xlim(min(xs_all)-pad, max(xs_all)+pad)
    ax.set_ylim(min(ys_all)-pad, max(ys_all)+pad)
    plt.tight_layout()
    plt.show()

# =========================
# Aide affichage "bien propre"
# =========================
def fmt_edge(u, v):
    return f"({u}-{v})"

def fmt_term(u, v):
    return f"|{coords[u][0]}-{coords[v][0]}|+|{coords[u][1]}-{coords[v][1]}|"

def print_pair_line(it_idx, i, j, A, B, C, D, before, after, delta, best_so_far=False):
    marque = " ✅" if best_so_far else ""
    print(
        f"  [#{it_idx:02d}] (i={i}, j={j}) "
        f"coupe {fmt_edge(A,B)} & {fmt_edge(C,D)}  (avant: {fmt_term(A,B)} + {fmt_term(C,D)} = {before}) | "
        f"ajoute {fmt_edge(A,C)} & {fmt_edge(B,D)}  (après: {fmt_term(A,C)} + {fmt_term(B,D)} = {after}) | "
        f"Δ = {after - before:+d}{marque}"
    )

# =========================
# Une génération (best-improvement) ou descente complète
# =========================
def one_generation_best_improvement(tour, gen_idx=1):
    """Teste toutes les paires valides, choisit le meilleur Δ. Renvoie (improved, new_tour, info)."""
    n = len(tour)
    best = None
    best_delta = 0
    best_pair = None
    best_ABCD = None
    best_before_after = None

    print(f"\n>>> Génération {gen_idx} — on teste toutes les paires 2-opt valides")
    it_idx = 0
    for (i, j) in valid_two_opt_pairs(n):
        it_idx += 1
        delta, (A, B, C, D), before, after = two_opt_delta(tour, i, j)
        # affichage détaillé des arêtes comparées
        print_pair_line(it_idx, i, j, A, B, C, D, before, after, delta,
                        best_so_far=(best_pair == (i, j) and delta < best_delta))
        # mise à jour du meilleur
        if best is None or delta < best_delta:
            best = two_opt_apply(tour, i, j)
            best_delta = delta
            best_pair = (i, j)
            best_ABCD = (A, B, C, D)
            best_before_after = (before, after)

    if best is not None and best_delta < 0:
        # préparer diff visuelle
        prev_edges = edge_set(tour)
        new_edges = edge_set(best)
        added = new_edges - prev_edges
        removed = prev_edges - new_edges
        info = {
            "pair": best_pair,
            "ABCD": best_ABCD,
            "delta": best_delta,
            "before_after": best_before_after,
            "added": added,
            "removed": removed,
        }
        # résumé de la génération
        (A, B, C, D) = best_ABCD
        before, after = best_before_after
        print(f"\n>>> Choix génération {gen_idx}: (i,j) = {best_pair}  "
              f"— coupe {fmt_edge(A,B)} & {fmt_edge(C,D)}  → ajoute {fmt_edge(A,C)} & {fmt_edge(B,D)}")
        print(f"    Coût local arêtes: avant={before}, après={after}, Δ={best_delta:+d}")
        return True, best, info
    else:
        print(f"\n>>> Génération {gen_idx}: aucune paire 2-opt n'améliore (Δ_min = {best_delta:+d}) → arrêt.")
        return False, tour, None

def hill_climb_2opt(tour, full_descent=True):
    """Si full_descent=False, ne fait qu'une seule génération. Sinon, répète jusqu'au blocage."""
    current = tour[:]
    it = 0
    while True:
        it += 1
        cost_before = cycle_cost(current)
        print(f"\n--- État avant génération {it} — tour = {current + [current[0]]} | coût = {cost_before}")
        improved, new_tour, info = one_generation_best_improvement(current, gen_idx=it)

        if not improved:
            print("=> Aucune amélioration : arrêt (optimum local).")
            break

        # tracé avant/après de cette génération
        title_before = f"Avant (gen {it}) — coût = {cost_before}"
        plot_cycle(current, title_before)

        title_after = (f"Après (gen {it}) — coût = {cycle_cost(new_tour)}  "
                       f"| Δ={info['delta']}  "
                       f"| coupé ({info['ABCD'][0]}-{info['ABCD'][1]}) & ({info['ABCD'][2]}-{info['ABCD'][3]})")
        # surligner arêtes ajoutées/retirées
        plot_cycle(new_tour, title_after, highlight_diff=(info["added"], info["removed"]))

        current = new_tour
        if not full_descent:
            # on ne fait qu'une génération si demandé
            break

    return current

# =========================
# Exécution
# =========================
if __name__ == "__main__":
    base_cycle = [1, 2, 3, 4, 5]
    print(f"Chemin initial : {base_cycle + [base_cycle[0]]}  |  coût = {cycle_cost(base_cycle)}")

    print("\n=== Mode 1 : une seule génération (best-improvement) ===")
    tour_after_one = hill_climb_2opt(base_cycle[:], full_descent=False)
    print(f"Résultat après 1 génération : {tour_after_one + [tour_after_one[0]]}  |  coût = {cycle_cost(tour_after_one)}")

    print("\n=== Mode 2 : descente complète (répéter jusqu'au blocage) ===")
    tour_final = hill_climb_2opt(base_cycle[:], full_descent=True)
    print(f"Solution finale : {tour_final + [tour_final[0]]}  |  coût = {cZAycle_cost(tour_final)}")
