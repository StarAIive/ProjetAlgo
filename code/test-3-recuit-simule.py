# =========================
# Simulated Annealing (Manhattan, 2-opt) — version "comme ton HC"
# =========================
import math
import random
import matplotlib.pyplot as plt

# -------------------------
# Données (identiques à ton HC)
# -------------------------
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
    """tour : liste SANS le dernier retour (ex: [1,2,3,4,5]).
       On boucle automatiquement (i -> i+1, et dernier -> premier)."""
    n = len(tour)
    total = 0
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        total += manhattan(a, b)
    return total

# -------------------------
# 2-opt (mêmes conventions que ton HC)
# -------------------------
def valid_two_opt_pairs(n):
    """Indices i<j avec j - i > 1 et (i=0, j=n-1) exclu (arête de retour)."""
    for i in range(n - 2):
        for j in range(i + 2, n - 1):
            yield i, j

def two_opt_apply(tour, i, j):
    """Inverse le segment (i+1 .. j) inclus."""
    new_tour = tour[:]
    new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
    return new_tour

def random_two_opt_move(n):
    """Tire au hasard une paire (i,j) valide."""
    pairs = list(valid_two_opt_pairs(n))
    return random.choice(pairs)

# -------------------------
# Visualisation (même style que ton HC)
# -------------------------
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

# -------------------------
# Recuit simulé
# -------------------------
def simulated_annealing_2opt(
    start_tour,
    T0=20.0,
    alpha=0.8,
    moves_per_T=6,
    T_min=0.2,
    seed=7,
    plot_first_k_accepts=3
):
    """Recuit simulé pas-à-pas (log + quelques tracés avant/après).
       - start_tour : ex. [1,2,3,4,5] (sans retour)
       - T0, alpha, moves_per_T, T_min : paramètres du recuit
       - plot_first_k_accepts : nombre d'acceptations pour lesquelles on trace avant/après
    """
    random.seed(seed)

    tour = start_tour[:]
    n = len(tour)
    cur_cost = cycle_cost(tour)
    best_tour = tour[:]
    best_cost = cur_cost

    logs = []  # liste de dicts pour chaque tentative de mouvement
    step = 0
    T = T0
    accepted_count = 0

    # Option : tracer l'état initial
    plot_cycle(tour, f"État initial — coût = {cur_cost}")

    while T > T_min:
        for _ in range(moves_per_T):
            step += 1

            # Tirer un mouvement 2-opt au hasard
            i, j = random_two_opt_move(n)
            cand = two_opt_apply(tour, i, j)
            cand_cost = cycle_cost(cand)
            delta = cand_cost - cur_cost

            if delta < 0:
                accept_prob = 1.0
                accepted = True
            else:
                accept_prob = math.exp(-delta / T)
                accepted = random.random() < accept_prob

            # Logging
            logs.append({
                "step": step,
                "T": T,
                "i": i, "j": j,
                "current_cost": cur_cost,
                "candidate_cost": cand_cost,
                "delta": delta,
                "accept_prob": accept_prob,
                "accepted": accepted,
                "best_cost": best_cost
            })

            # Affichage console clair
            print(f"[it {step:3d}] T={T:6.3f} | (i={i}, j={j}) | Δ={delta:+3d} | "
                  f"P={accept_prob:6.3f} | {'ACCEPTE' if accepted else 'refuse'} | "
                  f"cost={cur_cost:2d} -> {cand_cost:2d} | best={best_cost}")

            # Si on accepte, on met à jour
            if accepted:
                # Tracé avant/après pour les k premières acceptations
                if accepted_count < plot_first_k_accepts:
                    before_edges = edge_set(tour)
                    after_edges  = edge_set(cand)
                    added = after_edges - before_edges
                    removed = before_edges - after_edges
                    plot_cycle(tour, f"Avant (acceptation {accepted_count+1}) — coût = {cur_cost}")
                    plot_cycle(cand,
                               f"Après (acceptation {accepted_count+1}) — coût = {cand_cost} | Δ={delta} | (i={i}, j={j})",
                               highlight_diff=(added, removed))

                tour = cand
                cur_cost = cand_cost
                accepted_count += 1

                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_tour = tour[:]

        # refroidissement
        T *= alpha

    return best_tour, best_cost, logs

# -------------------------
# Run principal (exécution démo)
# -------------------------
if __name__ == "__main__":
    base_cycle = [1, 2, 3, 4, 5]
    print(f"Chemin initial : {base_cycle + [base_cycle[0]]}  |  coût = {cycle_cost(base_cycle)}")

    best_tour, best_cost, logs = simulated_annealing_2opt(
        base_cycle,
        T0=20.0, alpha=0.8, moves_per_T=6, T_min=0.2,
        seed=7,                 # pour rejouer la même trajectoire
        plot_first_k_accepts=3  # trace seulement les 3 premières acceptations
    )

    print("\n=== Résultat final ===")
    print(f"Meilleure tournée : {best_tour + [best_tour[0]]}")
    print(f"Coût (Manhattan)   : {best_cost}")

    # Courbes coût & température
    steps = [log["step"] for log in logs]
    cur_best = []
    temps = []
    best_so_far = float("inf")
    for log in logs:
        best_so_far = min(best_so_far, log["best_cost"])
        cur_best.append(best_so_far)
        temps.append(log["T"])

    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, [log["candidate_cost"] for log in logs], label="Coût candidat (au fil des essais)")
    plt.plot(steps, cur_best, label="Meilleur coût (cumulé)")
    plt.xlabel("Itération")
    plt.ylabel("Coût (distance totale)")
    plt.title("Évolution des coûts")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, temps, label="Température")
    plt.xlabel("Itération")
    plt.ylabel("Température")
    plt.title("Refroidissement (T)")
    plt.legend()
    plt.tight_layout()
    plt.show()# =========================
# Simulated Annealing (Manhattan, 2-opt) — version "comme ton HC"
# avec affichage des arêtes comparées à chaque itération
# =========================
import math
import random
import matplotlib.pyplot as plt

# -------------------------
# Données (identiques à ton HC)
# -------------------------
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
    """tour : liste SANS le dernier retour (ex: [1,2,3,4,5]).
       On boucle automatiquement (i -> i+1, et dernier -> premier)."""
    n = len(tour)
    total = 0
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        total += manhattan(a, b)
    return total

# -------------------------
# Helpers d'affichage d'arêtes
# -------------------------
def fmt_edge(u, v):
    return f"({u}-{v})"

def fmt_term(u, v):
    return f"|{coords[u][0]}-{coords[v][0]}|+|{coords[u][1]}-{coords[v][1]}|"

# -------------------------
# 2-opt (mêmes conventions que ton HC)
# -------------------------
def valid_two_opt_pairs(n):
    """Indices i<j avec j - i > 1 et (i=0, j=n-1) exclu (arête de retour)."""
    for i in range(n - 2):
        for j in range(i + 2, n - 1):
            yield i, j

def two_opt_apply(tour, i, j):
    """Inverse le segment (i+1 .. j) inclus."""
    new_tour = tour[:]
    new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
    return new_tour

def random_two_opt_move(n):
    """Tire au hasard une paire (i,j) valide."""
    pairs = list(valid_two_opt_pairs(n))
    return random.choice(pairs)

def two_opt_ABCD(tour, i, j):
    """Retourne les sommets A-B (coupure 1) et C-D (coupure 2) du 2-opt."""
    n = len(tour)
    A = tour[i]
    B = tour[(i+1) % n]
    C = tour[j]
    D = tour[(j+1) % n]
    return A, B, C, D

# -------------------------
# Visualisation (même style que ton HC)
# -------------------------
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

# -------------------------
# Recuit simulé — LOG détaillé par itération
# -------------------------
def simulated_annealing_2opt(
    start_tour,
    T0=20.0,
    alpha=0.8,
    moves_per_T=6,
    T_min=0.2,
    seed=7,
    plot_first_k_accepts=3
):
    """Recuit simulé pas-à-pas (log + quelques tracés avant/après).
       - start_tour : ex. [1,2,3,4,5] (sans retour)
       - T0, alpha, moves_per_T, T_min : paramètres du recuit
       - plot_first_k_accepts : nombre d'acceptations pour lesquelles on trace avant/après
    """
    random.seed(seed)

    tour = start_tour[:]
    n = len(tour)
    cur_cost = cycle_cost(tour)
    best_tour = tour[:]
    best_cost = cur_cost

    step = 0
    T = T0
    accepted_count = 0

    # Option : tracer l'état initial
    plot_cycle(tour, f"État initial — coût = {cur_cost}")

    while T > T_min:
        # Une "génération" = moves_per_T tirages à température constante
        for _ in range(moves_per_T):
            step += 1

            # Tirer un mouvement 2-opt au hasard
            i, j = random_two_opt_move(n)
            A, B, C, D = two_opt_ABCD(tour, i, j)

            # Coût local avant/après (pour les arêtes concernées)
            before_local = manhattan(A, B) + manhattan(C, D)       # arêtes supprimées
            after_local  = manhattan(A, C) + manhattan(B, D)       # arêtes ajoutées
            delta_local  = after_local - before_local

            cand = two_opt_apply(tour, i, j)
            cand_cost = cycle_cost(cand)
            delta = cand_cost - cur_cost

            # Probabilité d'acceptation (Δ>0) ou amélioration directe (Δ<0)
            if delta < 0:
                accept_prob = 1.0
                reason = f"amélioration (Δ={delta})"
                accepted = True
            else:
                accept_prob = math.exp(-delta / T)
                r = random.random()
                if r < accept_prob:
                    reason = (f"acceptée malgré dégradation (Δ=+{delta}) "
                              f"car tirage r={r:.3f} < P={accept_prob:.3f}")
                    accepted = True
                else:
                    reason = (f"refusée (Δ=+{delta}) car tirage r={r:.3f} ≥ P={accept_prob:.3f}")
                    accepted = False

            # --- LOG ULTRA-LISIBLE PAR ITÉRATION ---
            print(
                f"[it {step:3d}] T={T:6.3f} | (i={i}, j={j}) | "
                f"coupe {fmt_edge(A,B)} & {fmt_edge(C,D)} "
                f"(avant: {fmt_term(A,B)} + {fmt_term(C,D)} = {before_local}) | "
                f"ajoute {fmt_edge(A,C)} & {fmt_edge(B,D)} "
                f"(après: {fmt_term(A,C)} + {fmt_term(B,D)} = {after_local}) | "
                f"Δ_local={delta_local:+d} | Δ_cycle={delta:+d} | "
                f"P={accept_prob:6.3f} | {'ACCEPTE' if accepted else 'refuse'} | "
                f"cost={cur_cost:2d} -> {cand_cost:2d} | {reason}"
            )

            # Si on accepte, on met à jour (et on peut tracer)
            if accepted:
                if accepted_count < plot_first_k_accepts:
                    before_edges = edge_set(tour)
                    after_edges  = edge_set(cand)
                    added = after_edges - before_edges
                    removed = before_edges - after_edges
                    plot_cycle(tour, f"Avant (acceptation {accepted_count+1}) — coût = {cur_cost}")
                    plot_cycle(
                        cand,
                        (f"Après (acceptation {accepted_count+1}) — coût = {cand_cost} | "
                         f"Δ={delta} | (i={i}, j={j})"),
                        highlight_diff=(added, removed)
                    )

                tour = cand
                cur_cost = cand_cost
                accepted_count += 1

                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_tour = tour[:]

        # refroidissement
        T *= alpha

    return best_tour, best_cost

# -------------------------
# Run principal (exécution démo)
# -------------------------
if __name__ == "__main__":
    base_cycle = [1, 2, 3, 4, 5]
    print(f"Chemin initial : {base_cycle + [base_cycle[0]]}  |  coût = {cycle_cost(base_cycle)}")

    best_tour, best_cost = simulated_annealing_2opt(
        base_cycle,
        T0=20.0, alpha=0.8, moves_per_T=6, T_min=0.2,
        seed=7,                 # pour rejouer la même trajectoire
        plot_first_k_accepts=3  # trace seulement les 3 premières acceptations
    )

    print("\n=== Résultat final ===")
    print(f"Meilleure tournée : {best_tour + [best_tour[0]]}")
    print(f"Coût (Manhattan)   : {best_cost}")

