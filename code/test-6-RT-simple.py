# ===========================================
# Tabu Search (débutant) — TSP Manhattan, 2-opt
# Affiche la mémoire tabou à chaque itération
# Instance : 1=(3,8) 2=(7,0) 3=(0,2) 4=(0,5) 5=(8,4)
# Tour initial : [1,2,3,4,5] (retour implicite à 1)
# ===========================================

# ----- Données
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
    return abs(xa - xb) + abs(ya - yb)  # pas de diagonales

# ----- Coût d'une tournée (liste sans le retour)
def tour_cost(tour):
    n = len(tour)
    c = 0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        c += manhattan(u, v)
    return c

# ----- Arête non orientée (pour la table tabou)
def edge_key(u, v):
    return tuple(sorted((u, v)))

# ----- Application d'un move 2-opt et arêtes AJOUTÉES
def two_opt_apply(tour, i, j):
    """
    Applique 2-opt sur le segment [i..j] (1 <= i < j < n), retourne :
    - new_tour : la tournée modifiée
    - added    : les 2 arêtes nouvellement ajoutées par ce 2-opt
    Rappel : on garde le sommet tour[0] (client 1) fixe.
    """
    n = len(tour)
    # Avant 2-opt, les arêtes "externes" sont (a,b) et (c,d)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]

    # Après 2-opt, on ajoute (a,c) et (b,d)
    added = [edge_key(a, c), edge_key(b, d)]

    new_tour = tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
    return new_tour, added

# ----- Génération de tout le voisinage 2-opt
def generate_two_opt_moves(tour):
    """
    Génère tous les couples (i, j) valides en gardant tour[0] fixe.
    On évite :
      - (j == i+1) (move trivial)
      - (i == 1 and j == n-1) (inversion totale)
    """
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):  # j < n-1 pour éviter l'inversion totale
            if j == i + 1:
                continue
            yield (i, j)

# ----- Petite fonction d'affichage de la mémoire tabou
def print_tabu_memory(Tabu, it, prefix="[Tabu]"):
    # On n'affiche que les arêtes dont l'expiration est > it (donc encore actives)
    active = [(e, Tabu[e] - it) for e in Tabu if Tabu[e] > it]
    # Tri pour lisibilité : par durée restante décroissante
    active.sort(key=lambda x: -x[1])
    if not active:
        print(f"{prefix} (it={it}) : ∅ (vide)")
        return
    items = ", ".join([f"{e} (reste {remain})" for e, remain in active])
    print(f"{prefix} (it={it}) : {items}")

# ----- Recherche Tabou suivant ton pseudo-code, avec affichage
def tabu_search_simple(tour_init, K=100, tenure=5, verbose=True):
    # S ← solution initiale
    S = tour_init[:]
    # S* ← S
    S_best = S[:]
    C_best = tour_cost(S_best)

    # Tabu ← ∅  (dictionnaire : edge -> itération d'expiration)
    Tabu = {}

    if verbose:
        print(f"[INIT] S={S + [S[0]]}  C(S)={C_best}")
        print_tabu_memory(Tabu, it=0)

    for it in range(1, K + 1):
        # N ← générer_voisinage_2opt(S)
        moves = list(generate_two_opt_moves(S))

        best_v = None
        best_v_cost = None
        best_v_added = None

        # N_adm ← {v ∈ N | mouvement(v) non tabou OU aspiration(v, S*)}
        for (i, j) in moves:
            cand, added = two_opt_apply(S, i, j)
            c = tour_cost(cand)

            # Mouvement tabou si au moins une arête ajoutée est encore taboue
            is_tabu = any((e in Tabu and Tabu[e] > it) for e in added)
            # Aspiration si améliore le meilleur global
            aspiration = (c < C_best)

            if (not is_tabu) or aspiration:
                if best_v is None or c < best_v_cost:
                    best_v = cand
                    best_v_cost = c
                    best_v_added = added

        # Si aucun voisin admissible (rare ici), on s'arrête
        if best_v is None:
            if verbose:
                print(f"[Iter {it}] Aucun voisin admissible — arrêt.")
            break

        # appliquer mouvement v_best → S
        S = best_v

        # maj Tabu avec les arêtes AJOUTÉES (tenure ℓ)
        for e in best_v_added:
            Tabu[e] = it + tenure

        # Affichage demandé : move choisi et mémoire tabou courante
        if verbose:
            print(f"[Iter {it}] move choisi : ajoute {best_v_added}  | C(S)={best_v_cost}")
            print_tabu_memory(Tabu, it)

        # si C(S) < C(S*) alors S* ← S
        C_S = best_v_cost
        if C_S < C_best:
            S_best = S[:]
            C_best = C_S
            if verbose:
                print(f"[Iter {it}] NEW BEST  C(S*)={C_best}  S*={S_best + [S_best[0]]}")
        else:
            if verbose and it % 10 == 0:
                print(f"[Iter {it}] cur={C_S}  best={C_best}")

    return S_best, C_best

# ----- Exécution sur l'instance
if __name__ == "__main__":
    S0 = [1, 2, 3, 4, 5]  # retour implicite à 1
    S_star, C_star = tabu_search_simple(S0, K=100, tenure=5, verbose=True)

    print("\n=== Résultat final ===")
    print("Meilleure tournée :", S_star + [S_star[0]])
    print("Coût Manhattan    :", C_star)
