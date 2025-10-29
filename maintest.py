import os
import glob
import math
import matplotlib.pyplot as plt
import random
import time
import tracemalloc

choix = None
while choix != 0 and choix != 1:
    try:
        print("Choisissez le mode de test :")
        print("0 -> Tester le code de base (VRP standard)")
        print("1 -> Tester le code avec 2 contraintes supplémentaires")
        choix = int(input("Votre choix (0 ou 1) : "))
        if choix != 0 and choix != 1:
            print("Erreur : veuillez entrer 0 ou 1.")
    except ValueError:
        print(" Entrée invalide : veuillez entrer un nombre entier (0 ou 1).")

print(f"Mode sélectionné : {choix}")

if choix == 0:
    dossier = "data"
    fichiers_vrp = glob.glob(os.path.join(dossier, "*.vrp"))
    if not fichiers_vrp:
        print("Aucun fichier .vrp trouvé dans le dossier data/")
        raise SystemExit

    print("Fichiers .vrp disponibles :")
    idx = 1
    for fp in fichiers_vrp:
        print(f"{idx} - {os.path.basename(fp)}")
        idx += 1

    while True:
        try:
            k = int(input(f"\nChoisissez un fichier (1-{len(fichiers_vrp)}) : "))
            if 1 <= k <= len(fichiers_vrp):
                choix_fichier = fichiers_vrp[k - 1]
                nom_fichier = os.path.basename(choix_fichier)
                break
            else:
                print(f"Erreur : veuillez entrer un nombre entre 1 et {len(fichiers_vrp)}")
        except ValueError:
            print("Erreur : veuillez entrer un nombre valide")
    print(f"\nFichier sélectionné : {nom_fichier}")
else:
    print("Mode 1 (contraintes supplémentaires) : non implémenté pour l’instant.")
    raise SystemExit

def lire_coordonnees(fichier):
    coords = {}
    with open(fichier, "r") as f:
        lignes = f.readlines()
    n = len(lignes)
    i = 0
    while i < n and "NODE_COORD_SECTION" not in lignes[i]:
        i += 1
    if i == n:
        return coords
    i += 1
    while i < n and "DEMAND_SECTION" not in lignes[i] and "DEPOT_SECTION" not in lignes[i] and "EOF" not in lignes[i]:
        parts = lignes[i].split()
        if len(parts) >= 3:
            numero = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coords[numero] = (x, y)
        i += 1
    return coords

def lire_depots(fichier):
    depots = []
    with open(fichier, "r") as f:
        lignes = f.readlines()
    n = len(lignes)
    i = 0
    while i < n and "DEPOT_SECTION" not in lignes[i]:
        i += 1
    if i == n:
        return depots
    i += 1
    while i < n:
        line = lignes[i].strip()
        if line == "-1" or line == "EOF" or line == "":
            break
        depots.append(int(line.split()[0]))
        i += 1
    return depots

def lire_demandes(fichier):
    demandes = {}
    with open(fichier, "r") as f:
        lignes = f.readlines()
    n = len(lignes)
    i = 0
    while i < n and "DEMAND_SECTION" not in lignes[i]:
        i += 1
    if i == n:
        return demandes
    i += 1
    while i < n:
        line = lignes[i].strip()
        if line == "-1" or line == "EOF" or line.startswith("DEPOT_SECTION"):
            break
        parts = line.split()
        if len(parts) >= 2:
            numero = int(parts[0])
            demande = int(parts[1])
            demandes[numero] = demande
        i += 1
    return demandes

def lire_capacite(fichier):
    with open(fichier, "r") as f:
        lignes = f.readlines()
    for line in lignes:
        if "CAPACITY" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[1].strip())
                except ValueError:
                    return None
    return None

def solution_initiale(fichier):
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)
    demandes = lire_demandes(fichier)
    capacite = lire_capacite(fichier)
    if not depots:
        print("Aucun dépôt trouvé.")
        return []
    depot = depots[0]

    clients = []
    for i in coords.keys():
        est_depot = False
        for d in depots:
            if i == d:
                est_depot = True
                break
        if not est_depot:
            clients.append(i)

    tour = [depot]
    p = 0
    demande = 0
    while p < len(clients):
        if clients[p] in demandes:
            demande += demandes[clients[p]]
            if capacite is not None and demande > capacite:
                tour.append(depot)
                demande = 0
            else:
                tour.append(clients[p])
                p += 1
    tour.append(depot)
    routes = []
    courants = []
    i = 0
    n = len(tour)
    while i < n:
        v = tour[i]
        if v == depot:
            if len(courants) > 0:
                # on close la route: [depot] + clients + [depot]
                route = [depot]
                j = 0
                while j < len(courants):
                    route.append(courants[j])
                    j += 1
                route.append(depot)
                routes.append(route)
                courants = []
            # sinon: dépôt isolé -> on ignore (c'est juste une frontière)
        else:
            courants.append(v)
        i += 1
    return routes

def cout_total(routes, coords, metric="euclidienne"):
    total = 0.0
    for route in routes:
        if not route or len(route) < 2:
            continue
        i = 0
        while i < len(route) - 1:
            u = route[i]
            v = route[i + 1]
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            if metric == "manhattan":
                dist = abs(x1 - x2) + abs(y1 - y2)
            else: 
                dist = math.hypot(x1 - x2, y1 - y2)
            total += dist
            i += 1
    return total

def selection_op(poids_destruction, poids_reparation):
    """
    Sélectionne un opérateur de destruction et un opérateur de réparation 
    via un tirage à la roulette pondérée.
    
    Paramètres
    ----------
    poids_destruction : dict
        Exemple : {"random": 1, "worst": 1}
    poids_reparation : dict
        Exemple : {"greedy": 1, "best": 1}

    Retour
    ------
    op_destruct : str
    op_repare : str
    """
    
    # Générer une probabilité pour chaque opérateur de destruction à partir des poids
    total = sum(poids_destruction.values())
    proba_destruction = {}
    for k, v in poids_destruction.items():
        proba_destruction[k] = v / total


    # Tirer un opérateur de destruction selon la probabilité calculée
    op_destruct = random.choices(list(proba_destruction.keys()), weights=list(proba_destruction.values()))[0]

    # Générer une probabilité pour chaque opérateur de réparation à partir des poids
    total = sum(poids_reparation.values())
    proba_reparation = {}
    for k, v in poids_reparation.items():
        proba_reparation[k] = v / total

    # Tirer un opérateur de réparation selon la probabilité calculée
    op_repare = random.choices(list(proba_reparation.keys()), weights=list(proba_reparation.values()))[0]
    return op_destruct, op_repare

def worst_removal(routes, coords, q=2, metric="manhattan"):
    """
    Retire q clients impliqués dans les arêtes les plus coûteuses (Worst Removal - ALNS)
    en évitant les dépôts.
    """
    # Déterminer les dépôts présents dans la solution (tête/fin de chaque route)
    depot_ids = set()
    for route in routes:
        if route:
            depot_ids.add(route[0])
            depot_ids.add(route[-1])

    # Lister toutes les arêtes (u,v) en ignorant celles qui touchent un dépôt
    edges = []
    for route in routes:
        n = len(route)
        i = 0
        while i < n - 1:
            u = route[i]
            v = route[i + 1]
            if (u not in depot_ids) and (v not in depot_ids):
                edges.append((u, v))
            i += 1

    # Calculer le coût de chaque arête
    costs = []
    for (u, v) in edges:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        if metric == "manhattan":
            dist = abs(x1 - x2) + abs(y1 - y2)
        else:
            dist = math.hypot(x1 - x2, y1 - y2)
        costs.append((u, v, dist))

    # Trier par coût décroissant
    costs.sort(key=lambda t: t[2], reverse=True)

    # Sélectionner jusqu'à q clients (non dépôts), uniques, en parcourant les pires arêtes
    clients_a_retirer = []
    for (u, v, _) in costs:
        if u not in depot_ids and (u not in clients_a_retirer):
            clients_a_retirer.append(u)
            if len(clients_a_retirer) >= q:
                break
        if v not in depot_ids and (len(clients_a_retirer) < q) and (v not in clients_a_retirer):
            clients_a_retirer.append(v)
            if len(clients_a_retirer) >= q:
                break

    # Si q > clients disponibles, délimiter la liste
    if len(clients_a_retirer) > q:
        clients_a_retirer = clients_a_retirer[:q]

    # Retirer ces clients des routes (en conservant l'ordre et en préservant les dépôts)
    new_routes = []
    for route in routes:
        new_route = []
        for client in route:
            if (client in depot_ids) or (client not in clients_a_retirer):
                new_route.append(client)
        new_routes.append(new_route)

    return new_routes, clients_a_retirer

def delta_insertion(route, idx, node, coords, metric="manhattan"):
    """
    Δ = d(a,node) + d(node,b) - d(a,b), avec a = route[idx], b = route[idx+1]
    """
    a = route[idx]
    b = route[idx + 1]
    ax, ay = coords[a]; nx, ny = coords[node]; bx, by = coords[b]

    if metric == "manhattan":
        delta = (abs(ax - nx) + abs(ay - ny)) + (abs(nx - bx) + abs(ny - by)) - (abs(ax - bx) + abs(ay - by))
    else:
        delta = (math.hypot(ax - nx, ay - ny) + math.hypot(nx - bx, ny - by) - math.hypot(ax - bx, ay - by))

    return delta


def insertion_faisable(route, idx, node, demandes=None, capacite=None, contraintes=None):
    """
    Faisabilité minimale : capacité par route (les dépôts sont aux extrémités).
    """
    if demandes is not None and capacite is not None:
        charge = 0
        i = 1
        while i < len(route) - 1:  # ignorer dépôts
            client = route[i]
            charge += demandes.get(client, 0)
            i += 1
        charge += demandes.get(node, 0)
        if charge > capacite:
            return False

    # (extensions TW/compatibilité plus tard via `contraintes`)
    return True


def greedy_insertion(routes, node, coords, metric="manhattan",
                          demandes=None, capacite=None, contraintes=None):
    """
    Insère `node` à la meilleure position (Δ minimal) sur l’ensemble des routes.
    """
    delta_best = float("inf")
    choix_best = (None, None)  # (r_idx, idx)

    # Recherche du meilleur emplacement faisable
    r_idx = 0
    while r_idx < len(routes):
        route = routes[r_idx]
        i = 0
        while i < len(route) - 1:
            if insertion_faisable(route, i, node, demandes, capacite, contraintes):
                delta = delta_insertion(route, i, node, coords, metric)
                if delta < delta_best:
                    delta_best = delta
                    choix_best = (r_idx, i)
            i += 1
        r_idx += 1

    # Aucun placement faisable trouvé
    if choix_best == (None, None):
        return routes, float("inf"), False

    # Appliquer l’insertion
    r, i = choix_best
    routes[r].insert(i + 1, node)
    return routes, delta_best, True


def reparation_greedy(routes_partial, removed, coords, metric="manhattan",
                  demandes=None, capacite=None, contraintes=None, ordre="as_is"):
    """
    Réinsère toutes les villes de `removed` en mode Greedy (Δ minimal successif).
    """
    # Copie de travail
    routes_modifiees = []
    for rt in routes_partial:
        routes_modifiees.append(rt[:])

    delta_total = 0.0
    non_inseres = []

    ordre_effectif = list(removed)

    for u in ordre_effectif:
        routes_modifiees, delta, ok = greedy_insertion(
            routes_modifiees, u, coords, metric,
            demandes=demandes, capacite=capacite, contraintes=contraintes
        )
        if ok:
            delta_total += (0.0 if delta is None else delta)
        else:
            non_inseres.append(u)

    return routes_modifiees, non_inseres, delta_total

def best_reparation(routes_partial, removed, coords, metric="manhattan",
                demandes=None, capacite=None, contraintes=None, ordre="sorted_by_cost"):
    """
    Réinsère en triant les villes par coût d'insertion croissant (best insertion).
    """
    routes_modifiees = []
    for rt in routes_partial:
        routes_modifiees.append(rt[:])

    delta_total = 0.0
    non_inseres = []
    
    # Pour chaque ville à insérer
    for u in removed:
        best_cost = float("inf")
        best_pos = None
        
        # Chercher la meilleure position parmi toutes les routes
        for r_idx, route in enumerate(routes_modifiees):
            for i in range(len(route) - 1):
                if insertion_faisable(route, i, u, demandes, capacite, contraintes):
                    delta = delta_insertion(route, i, u, coords, metric)
                    if delta < best_cost:
                        best_cost = delta
                        best_pos = (r_idx, i)
        
        if best_pos is not None:
            r_idx, i = best_pos
            routes_modifiees[r_idx].insert(i + 1, u)
            delta_total += best_cost
        else:
            non_inseres.append(u)
    
    return routes_modifiees, non_inseres, delta_total

def acceptation_regle(delta, T, mode="sa", epsilon=0.0):
    """
    Décide si on accepte une solution candidate (delta = C(S') - C(S)).
    """
    # Toute amélioration est acceptée
    if delta < 0:
        return True

    # Modes d'acceptation
    if mode == "sa":
        # Garde-fou pour éviter division par 0 ou T très petit
        if T is None or T <= 1e-12:
            return False
        prob = math.exp(-delta / T)
        return random.random() < prob

    elif mode == "improve_only":
        return False

    elif mode == "threshold":
        return delta <= epsilon

    # Par défaut : refuser
    return False

def appli_acceptation(state, candidate, selected_ops, T, params, scores):
    """
    Applique acceptation/rejet + met à jour S/C, best global, T, et les crédits opérateurs.
    """
    C_cur = state["C"]
    C_new = candidate["C_new"]
    delta = C_new - C_cur

    # Décision d'acceptation
    accept = acceptation_regle(
        delta, 
        T, 
        mode=params.get("accept_mode", "sa"), 
        epsilon=params.get("epsilon", 0.0)
    )

    op_remove, op_insert = selected_ops

    if not accept:
        # Rejet : on crédite juste les 'uses'
        scores["remove"][op_remove]["uses"] += 1
        scores["insert"][op_insert]["uses"] += 1

        # Refroidissement éventuel
        if params.get("accept_mode", "sa") == "sa":
            T = params.get("alpha", 0.995) * T

        # Mise à jour par segment (incrément du compteur + éventuel update des poids)
        scores["iters_in_segment"] += 1
        if scores["iters_in_segment"] >= scores["segment_len"]:
            rho = scores["rho"]
            for fam in ["remove", "insert"]:
                for op, data in scores[fam].items():
                    w_old = data["weight"]
                    uses = max(1, data["uses"])
                    sc = data["score"]
                    data["weight"] = (1 - rho) * w_old + rho * (sc / uses)
                    data["score"] = 0.0
                    data["uses"] = 0
            scores["iters_in_segment"] = 0

        return state, T, scores, "rejected"

    # Accepté : on met à jour l'état courant
    new_state = {
        "S": candidate["S_new"],
        "C": C_new,
        "S_best": state["S_best"],
        "C_best": state["C_best"]
    }

    # Outcome + meilleur global éventuel
    if C_new < state["C_best"]:
        new_state["S_best"] = candidate["S_new"]
        new_state["C_best"] = C_new
        outcome = "best_global"
    elif delta < 0:
        outcome = "improve"
    else:
        outcome = "accepted_worse"

    # Créditer opérateurs (score + uses) — remove ET insert
    pi = scores["pi"]
    credit = pi.get(outcome, 0.0)
    scores["remove"][op_remove]["score"] += credit
    scores["remove"][op_remove]["uses"] += 1
    scores["insert"][op_insert]["score"] += credit
    scores["insert"][op_insert]["uses"] += 1

    # Refroidissement éventuel
    if params.get("accept_mode", "sa") == "sa":
        T = params.get("alpha", 0.995) * T

    # Mise à jour par segment (incrément du compteur + éventuel update des poids)
    scores["iters_in_segment"] += 1
    if scores["iters_in_segment"] >= scores["segment_len"]:
        rho = scores["rho"]
        for fam in ["remove", "insert"]:
            for op, data in scores[fam].items():
                w_old = data["weight"]
                uses = max(1, data["uses"])
                sc = data["score"]
                data["weight"] = (1 - rho) * w_old + rho * (sc / uses)
                data["score"] = 0.0
                data["uses"] = 0
        scores["iters_in_segment"] = 0

    return new_state, T, scores, outcome


def init_scores_et_param(remove_ops, insert_ops):
    """
    Initialise les structures de scores/poids + paramètres d’acceptation.
    """
    scores = {
        "remove": {},
        "insert": {},
        "pi": {  # barèmes ultra-agressifs pour favoriser les bonnes découvertes
            "best_global": 50.0,  # récompense maximale pour les meilleures solutions
            "improve": 25.0,      # récompense très élevée pour les améliorations
            "accepted_worse": 3.0 # récompense modérée pour diversification
        },
        "rho": 0.5,          # mise à jour ultra-agressive des poids
        "segment_len": 50,   # segments plus courts pour adaptation très rapide
        "iters_in_segment": 0
    }

    for op in remove_ops:
        scores["remove"][op] = {"score": 0.0, "uses": 0, "weight": 1.0}
    for op in insert_ops:
        scores["insert"][op] = {"score": 0.0, "uses": 0, "weight": 1.0}

    params = {
        "accept_mode": "sa",   # 'sa' | 'improve_only' | 'threshold'
        "alpha": 0.9995,       # refroidissement ultra-lent pour exploration prolongée
        "epsilon": 0.0         # seuil pour mode 'threshold'
    }

    return scores, params


def alns_iteration(state, coords, metric, T, params, scores,
                   op_remove, op_repair, remove_func, repair_func, q_remove,
                   demandes=None, capacite=None, contraintes=None):
    """
    Orchestration d'une itération ALNS :
    - destruction -> réparation -> coût -> acceptation -> MAJ T/poids
    `remove_func` et `repair_func` sont des fonctions OPÉRATEUR-SPÉCIFIQUES.
    """

    # Destruction
    routes_partial, removed = remove_func(
        state["S"], coords, q_remove, metric
    )

    # Réparation 
    routes_candidate, non_inseres, _ = repair_func(
        routes_partial, removed, coords, metric,
        demandes=demandes, capacite=capacite, contraintes=contraintes
    )

    if non_inseres:
        return state, T, scores, "rejected"

    #Coût candidat
    C_new = cout_total(routes_candidate, coords, metric)

    # Candidate dict
    candidate = {"S_new": routes_candidate, "C_new": C_new}

    # Acceptation + MAJ T/scores/poids
    new_state, T, scores, outcome = appli_acceptation(
        state, candidate, (op_remove, op_repair), T, params, scores
    )

    return new_state, T, scores, outcome

def refroidissement(T, params):
    """
    Refroidissement simple : T <- alpha * T (simulated annealing)
    """
    if params.get("accept_mode", "sa") == "sa":
        T = max(1e-12, params.get("alpha", 0.995) * T)
    return T

def random_removal(routes, coords, q=2, metric="manhattan"):
    """
    Retire aléatoirement q clients (hors dépôts) de l'ensemble des routes.
    Retourne (routes_partial, removed).
    """
    # Collecter dépôts (tête/fin de chaque route)
    depots = set()
    for r in routes:
        if r:
            depots.add(r[0]); depots.add(r[-1])

    # Lister tous les clients retirables (non dépôts)
    pool = []
    r_idx = 0
    while r_idx < len(routes):
        route = routes[r_idx]
        i = 1
        while i < len(route) - 1:
            pool.append(route[i])
            i += 1
        r_idx += 1

    if not pool:
        routes_copy = []
        for rt in routes:
            routes_copy.append(rt[:])

        return routes_copy, []

    # Échantillonner sans remise au plus q
    k = min(q, len(pool))
    removed = random.sample(pool, k)

    # Construire les nouvelles routes (on enlève removed, on garde dépôts)
    new_routes = []
    r_idx = 0
    while r_idx < len(routes):
        route = routes[r_idx]
        new_r = []
        j = 0
        while j < len(route):
            v = route[j]
            if (v in depots) or (v not in removed):
                new_r.append(v)
            j += 1
        new_routes.append(new_r)
        r_idx += 1

    return new_routes, removed

def shaw_removal(routes, coords, q=2, metric="euclidienne"):
    """
    Shaw removal : retire q clients similaires (proches géographiquement).
    Plus efficace pour les grandes instances.
    """
    # Collecter dépôts
    depots = set()
    for r in routes:
        if r:
            depots.add(r[0]); depots.add(r[-1])

    # Lister tous les clients retirables (non dépôts)
    pool = []
    for route in routes:
        for i in range(1, len(route) - 1):
            if route[i] not in depots:
                pool.append(route[i])

    if not pool or q <= 0:
        return [rt[:] for rt in routes], []

    # Choisir un client de départ aléatoirement
    seed_client = random.choice(pool)
    removed = [seed_client]
    
    # Ajouter les q-1 clients les plus proches du seed
    while len(removed) < q and len(removed) < len(pool):
        best_dist = float("inf")
        best_client = None
        
        for client in pool:
            if client not in removed:
                # Distance minimale au cluster déjà sélectionné
                min_dist_to_cluster = min(
                    math.hypot(coords[client][0] - coords[selected][0],
                              coords[client][1] - coords[selected][1])
                    for selected in removed
                )
                
                if min_dist_to_cluster < best_dist:
                    best_dist = min_dist_to_cluster
                    best_client = client
        
        if best_client is not None:
            removed.append(best_client)
        else:
            break

    # Construire les nouvelles routes
    new_routes = []
    for route in routes:
        new_route = []
        for client in route:
            if (client in depots) or (client not in removed):
                new_route.append(client)
        new_routes.append(new_route)

    return new_routes, removed

def selection_de_scores(scores):
    """
    Construit les distributions de poids à partir de `scores` et sélectionne
    (op_remove, op_insert) par tirage pondéré.
    Si `selection_op(...)` existe déjà chez toi, tu peux la remplacer ici.
    """
    # Tables de poids
    w_remove = {}
    for op, data in scores["remove"].items():
        w_remove[op] = data["weight"]

    w_insert = {}
    for op, data in scores["insert"].items():
        w_insert[op] = data["weight"]


    # Tirage par roulette (pondéré)

    op_remove, op_insert = selection_op(w_remove, w_insert) 
    return op_remove, op_insert

def optimisation_2opt(routes, coords, metric="euclidienne"):
    """
    Applique l'optimisation 2-opt améliorée sur chaque route individuellement.
    """
    improved_routes = []
    
    for route in routes:
        if len(route) <= 4:  # route trop courte pour 2-opt efficace
            improved_routes.append(route[:])
            continue
            
        best_route = route[:]
        improved = True
        max_iterations = 5  # limite pour éviter la stagnation
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Calcul incrémental des gains pour éviter recalculs coûteux
            for i in range(1, len(best_route) - 2):
                for j in range(i + 2, min(i + 20, len(best_route) - 1)):  # limite voisinage
                    # Calcul du gain 2-opt directement
                    x1, y1 = coords[best_route[i-1]]
                    x2, y2 = coords[best_route[i]]
                    x3, y3 = coords[best_route[j]]
                    x4, y4 = coords[best_route[j+1]]
                    
                    if metric == "manhattan":
                        old_cost = (abs(x1-x2) + abs(y1-y2)) + (abs(x3-x4) + abs(y3-y4))
                        new_cost = (abs(x1-x3) + abs(y1-y3)) + (abs(x2-x4) + abs(y2-y4))
                    else:
                        old_cost = math.hypot(x1-x2, y1-y2) + math.hypot(x3-x4, y3-y4)
                        new_cost = math.hypot(x1-x3, y1-y3) + math.hypot(x2-x4, y2-y4)
                    
                    if new_cost < old_cost:  # Amélioration trouvée
                        # Appliquer le 2-opt
                        best_route[i:j+1] = reversed(best_route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
        
        improved_routes.append(best_route)
    
    return improved_routes

def alns(initial_routes, coords,
         metric="manhattan",
         n_iter=500,
         q_remove=2,
         demandes=None,
         capacite=None,
         contraintes=None,
         seed=None,
         log_every=50):
    """
    Lance l'ALNS sur une solution initiale.
    Retourne l'état final (incluant le meilleur global).
    """

    if seed is not None:
        random.seed(seed)

    # Opérateurs disponibles - ajout de diversité
    remove_ops = ["worst", "random", "shaw"]
    insert_ops = ["greedy", "best"]

    # Init scores & params (tes fonctions déjà corrigées)
    scores, params = init_scores_et_param(remove_ops, insert_ops)

    # Température initiale
    T = 200.0 

    # État courant
    S0 = []
    for rt in initial_routes:
        S0.append(rt[:])
    C0 = cout_total(S0, coords, metric)
    state = {
        "S": S0,
        "C": C0,
        "S_best": [],
    }
    for rt in S0:
        state["S_best"].append(rt[:])
    state["C_best"] = C0

    # Dictionnaires d’opérateurs -> fonctions
    remove_funcs = {
        "worst": lambda routes, coords, q, metric:
            worst_removal(routes, coords, q=q, metric=metric),
        "random": lambda routes, coords, q, metric:
            random_removal(routes, coords, q=q, metric=metric),
        "shaw": lambda routes, coords, q, metric:
            shaw_removal(routes, coords, q=q, metric=metric),
    }
    repair_funcs = {
        "greedy": lambda routes_partial, removed, coords, metric, **kw:
            reparation_greedy(routes_partial, removed, coords, metric, **kw),
        "best": lambda routes_partial, removed, coords, metric, **kw:
            best_reparation(routes_partial, removed, coords, metric, **kw),
    }

    # Boucle ALNS
    it = 1
    while it <= n_iter:
        # 1 Sélection opérateurs via poids
        op_remove, op_insert = selection_de_scores(scores)

        # 2 Récup fonctions
        remove_func = remove_funcs.get(op_remove)
        repair_func = repair_funcs.get(op_insert)
        if remove_func is None or repair_func is None:
            # Opérateur inconnu -> on passe (sécurité)
            it += 1
            continue

        # Une itération ALNS (ta fonction)
        state, T, scores, outcome = alns_iteration(
            state, coords, metric, T, params, scores,
            op_remove, op_insert, remove_func, repair_func, q_remove,
            demandes=demandes, capacite=capacite, contraintes=contraintes
        )

        # Optimisation locale périodique (2-opt toutes les 100 itérations)
        if it % 100 == 0:
            optimized_routes = optimisation_2opt(state["S"], coords, metric)
            optimized_cost = cout_total(optimized_routes, coords, metric)
            
            if optimized_cost < state["C"]:
                state["S"] = optimized_routes
                state["C"] = optimized_cost
                
                if optimized_cost < state["C_best"]:
                    S_best = []
                    for rt in optimized_routes:
                        S_best.append(rt[:])

                    state["S_best"] = S_best
                    state["C_best"] = optimized_cost

        # Logs
        if (log_every is not None) and (it % log_every == 0):
            print(f"[ALNS] it={it:5d} | outcome={outcome:15s} | C={state['C']:.2f} | C*={state['C_best']:.2f} | T={T:.4f}")

        it += 1

    # 6) Optimisation finale 2-opt
    final_optimized = optimisation_2opt(state["S_best"], coords, metric)
    final_cost = cout_total(final_optimized, coords, metric)
    
    if final_cost < state["C_best"]:
        state["S_best"] = final_optimized
        state["C_best"] = final_cost
        print(f"[POST-OPT] Amélioration finale: {final_cost:.2f}")

    # 7) Retour
    return state

def gap(cout, fichier_vrp):
    """
    Calcule l'écart (en %) entre un coût courant et le coût optimal indiqué
    dans le fichier .sol correspondant (même nom que le .vrp, extension .sol).
    Retourne un float (pourcentage) ou None si le .sol est introuvable ou illisible.
    """
    try:
        # Construire le chemin vers le fichier .sol
        nom_fichier = os.path.basename(fichier_vrp)  # ex: "A-n32-k5.vrp"
        base, _ = os.path.splitext(nom_fichier)       # ex: "A-n32-k5"
        sol_path = os.path.join("data", base + ".sol")  # ex: "data/A-n32-k5.sol"
        
        with open(sol_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Cost"):
                    # Extraire le nombre après "Cost"
                    # Format attendu: "Cost 784" ou "Cost: 784"
                    parts = line.replace(":", "").split()
                    if len(parts) >= 2:
                        try:
                            opt = float(parts[1])
                            if opt == 0:
                                return None
                            return 100.0 * (cout - opt) / opt
                        except ValueError:
                            continue
        return None
    except FileNotFoundError:
        print(f"Fichier .sol non trouvé: {sol_path}")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier .sol: {e}")
        return None

def tracer_vrp(fichier, routes=None, titre="Clients et Dépôts"):
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)

    # Séparer clients
    clients = []
    for i in coords.keys():
        est_depot = False
        for d in depots:
            if i == d:
                est_depot = True
                break
        if not est_depot:
            clients.append(i)

    # Points
    x_clients, y_clients = [], []
    for cid in clients:
        x, y = coords[cid]
        x_clients.append(x); y_clients.append(y)

    x_depots, y_depots = [], []
    for did in depots:
        x, y = coords[did]
        x_depots.append(x); y_depots.append(y)

    # Plot
    plt.figure()
    plt.scatter(x_clients, y_clients, c='blue', label='Clients')
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    # Labels
    for i, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, color="black")

    # Routes : couleur différente par route
    if routes is not None and len(routes) > 0:
        couleurs = plt.cm.tab20.colors  # palette de 20 couleurs
        for r_idx, route in enumerate(routes):
            col = couleurs[r_idx % len(couleurs)]
            label = f"Camion {r_idx + 1}"
            # On met le label uniquement sur la première arête de la route,
            premiere = True

            i = 0
            while i < len(route) - 1:
                x1, y1 = coords[route[i]]
                x2, y2 = coords[route[i + 1]]
                if premiere:
                    plt.plot([x1, x2], [y1, y2], '-', color=col, label=label)
                    premiere = False
                else:
                    plt.plot([x1, x2], [y1, y2], '-', color=col)
                i += 1

    plt.title(titre)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


coords = lire_coordonnees(choix_fichier)
routes = solution_initiale(choix_fichier)
tracer_vrp(choix_fichier, routes, titre="Solution Initiale VRP")
print("cout total de la solution initiale :", cout_total(routes, coords, metric="manhattan"))

# ALNS
debut = time.perf_counter()
tracemalloc.start()
state_final = alns(
    initial_routes=routes,
    coords=coords,
    metric="euclidienne",  # métrique euclidienne pour meilleurs résultats
    n_iter=20000, # beaucoup plus d'itérations pour grandes instances
    q_remove=6,   # destruction encore plus agressive
    demandes=lire_demandes(choix_fichier),
    capacite=lire_capacite(choix_fichier),
    contraintes=None,
    seed=42,
    log_every=2000
)

print("Coût final       :", state_final["C"])
print("Meilleur global :", state_final["C_best"])
print("Écart (%)    :", gap(state_final["C_best"], choix_fichier))

fin = time.perf_counter()
print("Temps d'exécution :", fin - debut, "secondes")
current, peak = tracemalloc.get_traced_memory()
print(f"Mémoire actuelle : {current / 1024:.2f} Ko")

tracer_vrp(choix_fichier, state_final["S_best"], titre="ALNS: meilleure solution")