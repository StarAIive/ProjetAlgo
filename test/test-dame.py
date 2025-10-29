import random
from typing import List, Tuple, Optional

State = List[int]

def conflicts(state: State) -> int:
    n = len(state)
    cnt = 0
    for c1 in range(n):
        r1 = state[c1]
        for c2 in range(c1 + 1, n):
            r2 = state[c2]
            if r1 == r2:
                cnt += 1  # même ligne
            elif abs(r1 - r2) == abs(c1 - c2):
                cnt += 1  # même diagonale
    return cnt

def neighbors(state: State) -> List[Tuple[State, Tuple[int,int]]]:
    """
    Génère tous les voisins en déplaçant chaque reine (colonne c) sur les autres lignes.
    Retourne (nouvel_etat, (colonne, nouvelle_ligne)).
    """
    n = len(state)
    res = []
    for c in range(n):
        current_r = state[c]
        for r in range(n):
            if r == current_r:
                continue
            new_state = state[:]
            new_state[c] = r
            res.append((new_state, (c, r)))
    return res

def argmin_states(states: List[Tuple[State, Tuple[int,int]]], key_fn) -> List[Tuple[State, Tuple[int,int]]]:
    """
    Renvoie la liste des états atteignant le coût minimal (utile pour casser les égalités au hasard).
    """
    best = None
    best_list = []
    for s, move in states:
        val = key_fn(s)
        if best is None or val < best:
            best = val
            best_list = [(s, move)]
        elif val == best:
            best_list.append((s, move))
    return best_list

def print_board(state: State) -> None:
    n = len(state)
    for r in range(n):
        row = []
        for c in range(n):
            row.append('Q' if state[c] == r else '.')
        print(' '.join(row))
    print()

def hill_climbing_queens(
    n: int = 4,
    seed: Optional[int] = 0,
    sideways_limit: int = 0,
    max_restarts: int = 10,
    verbose: bool = True
) -> Tuple[State, int, int, int]:
    """
    Hill Climbing pour N-Queens.
    - sideways_limit : nombre max de pas latéraux (coût identique) autorisés d'affilée
    - max_restarts   : nb max de redémarrages aléatoires si plateau/cul-de-sac
    Retourne (state, cost, steps_total, restarts_used).
    """
    rng = random.Random(seed)
    steps_total = 0
    restarts_used = 0

    while True:
        # État initial aléatoire : une reine par colonne, ligne aléatoire
        current = [rng.randrange(n) for _ in range(n)]
        current_cost = conflicts(current)
        sideways_left = sideways_limit

        if verbose:
            print(f"=== Nouveau départ (restart {restarts_used}) ===")
            print(f"État initial : {current} | Conflits = {current_cost}")

        improved = True
        while improved:
            improved = False
            neighs = neighbors(current)
            # Chercher les voisins avec le coût minimal
            best_list = argmin_states(neighs, key_fn=conflicts)
            # Choisir au hasard parmi les meilleurs (évite un biais d'ordre)
            candidate, move = rng.choice(best_list)
            cand_cost = conflicts(candidate)

            if cand_cost < current_cost:
                if verbose:
                    print(f"  Move colonne {move[0]} -> ligne {move[1]} | {current_cost} -> {cand_cost}")
                current, current_cost = candidate, cand_cost
                steps_total += 1
                improved = True
                sideways_left = sideways_limit  # on réinitialise les pas latéraux
            elif cand_cost == current_cost and sideways_left > 0:
                # Pas latéral (plateau)
                if verbose:
                    print(f"  Sideways colonne {move[0]} -> ligne {move[1]} | {current_cost} = {cand_cost} (reste {sideways_left-1})")
                current, current_cost = candidate, cand_cost
                sideways_left -= 1
                steps_total += 1
                improved = True
            else:
                # Aucun voisin strictement meilleur (ou plus de sideways autorisés) -> plateau/local optimum
                pass

            if current_cost == 0:
                if verbose:
                    print("  Solution atteinte !")
                print_board(current)
                return current, current_cost, steps_total, restarts_used

        # Ici : pas d'amélioration possible
        if verbose:
            print(f"Plateau/local optimum avec {current_cost} conflits.")
        if restarts_used >= max_restarts:
            if verbose:
                print("Arrêt : nombre maximal de redémarrages atteint.")
            print_board(current)
            return current, current_cost, steps_total, restarts_used
        restarts_used += 1
        # On repart d'un nouvel état aléatoire (random-restart)

if __name__ == "__main__":
    # Exemple d'exécution pour 4 reines
    state, cost, steps, restarts = hill_climbing_queens(
        n=4,
        seed=42,           # fixe la reproductibilité de la trace
        sideways_limit=5,  # autoriser quelques pas latéraux aide sur 4×4
        max_restarts=10,
        verbose=True
    )
    print(f"Résultat final : {state} | Conflits = {cost} | Pas = {steps} | Restarts = {restarts}")
