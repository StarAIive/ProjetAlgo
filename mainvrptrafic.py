import os
import glob
import math
import matplotlib.pyplot as plt
import random
import time
import tracemalloc
import vrplib
from datetime import datetime

# MENU PRINCIPAL

choix = None
dossier = "data"
fact = 1.0  # Facteur de trafic par défaut

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
    
    # Charger l'instance VRP avec vrplib
    instance = vrplib.read_instance(choix_fichier)
    print(f"Instance chargée : {instance['dimension']} nœuds, capacité {instance['capacity']}")
else:

    print("Mode 1 (contraintes supplémentaires) :")
    choix1 = None
    while choix1 != 0 and choix1 != 1 and choix1 != 2:
        try:
            print("Choisissez la contrainte :")
            print("0 -> Contraintes fenetre de temps")
            print("1 -> Contraintes facteur de traffic")
            print("2 -> Les deux contraintes")
            choix1 = int(input("Votre choix (0, 1 ou 2) : "))
            if choix1 != 0 and choix1 != 1 and choix1 != 2:
                print("Erreur : veuillez entrer 0, 1 ou 2.")

        except ValueError:
            print(" Entrée invalide : veuillez entrer un nombre entier (0, 1 ou 2).")

    print(f"Contrainte sélectionnée : {choix1}")
    choix2 = int(input("Choisissez le mode de trafic 0.Automatique, 1.Manuel : "))
    if choix2 == 0:
        """Retourne un facteur de trafic temporelle en fonction de l'heure de la journée"""
        now = datetime.now()
        h = now.hour
        if 7 <= h < 9:
            fact = 1.3  # Heure de pointe du matin
        elif 9 <= h < 16:
            fact = 1.0  # Heure normale de l'après-midi
        elif 17 <= h < 20:
            fact =  1.8  # Heure de pointe du soir
        else:
            fact = 0.8  # Autres heures
    else:
    
        """Retourne un facteur de trafic temporelle entré manuellement par l'utilisateur"""

        h = int(input("Entrez l'heure (0-23) : "))
        if 7 <= h < 9:
            fact = 1.3  # Heure de pointe du matin
        elif 9 <= h < 16:
            fact = 1.0  # Heure normale de l'après-midi
        elif 17 <= h < 20:
            fact = 1.8  # Heure de pointe du soir
        else:
            fact = 0.8  # Autres heures


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
        
    # Charger l'instance VRP avec vrplib
    instance = vrplib.read_instance(choix_fichier)
    print(f"Instance chargée : {instance['dimension']} nœuds, capacité {instance['capacity']}")

# Fonctions utilisant VRPLib remplacent les anciennes fonctions de lecture
def get_coords_dict(instance):
    """Convertit les coordonnées VRPLib (numpy array) en dictionnaire pour compatibilité"""
    coords = {}
    for i in range(instance['dimension']):
        coords[i] = (instance['node_coord'][i][0], instance['node_coord'][i][1])
    return coords

def get_demands_dict(instance):
    """Convertit les demandes VRPLib (numpy array) en dictionnaire pour compatibilité"""
    demands = {}
    for i in range(instance['dimension']):
        demands[i] = instance['demand'][i]
    return demands

def solution_initiale(instance):
    """
    Génère une solution initiale en utilisant une heuristique simple avec les données VRPLib.
    """
    depot = instance["depot"][0]  # Index du dépôt (généralement 0)
    demand = instance["demand"]   # Numpy array des demandes
    capacity = instance["capacity"]  # Capacité des véhicules
    n = instance["dimension"]     # Nombre de nœuds
    
    # Les clients sont tous les nœuds sauf le dépôt
    clients = []
    i = 0
    while i < n:
        if i != depot:
            clients.append(i)
        i += 1

    
    tour = [depot]
    p = 0
    demande = 0
    while p < len(clients):
        client_demande = demand[clients[p]]
        demande += client_demande
        if capacity is not None and demande > capacity:
            tour.append(depot)
            demande = client_demande  # Commencer nouvelle route avec ce client
            tour.append(clients[p])
            p += 1
        else:
            tour.append(clients[p])
            p += 1
    tour.append(depot)
    
    # Convertir le tour en routes
    routes = []
    courants = []
    i = 0
    n_tour = len(tour)
    while i < n_tour:
        v = tour[i]
        if v == depot:
            if len(courants) > 0:
                # Fermer la route: [depot] + clients + [depot]
                route = [depot]
                j = 0
                while j < len(courants):
                    route.append(courants[j])
                    j += 1
                route.append(depot)
                routes.append(route)
                courants = []
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
            total += dist * fact  # Appliquer le facteur de trafic réel
            i += 1
    return total

def selection_op(scores):
    """
    Sélectionne un opérateur de destruction et un opérateur de réparation 
    via un tirage à la roulette pondérée.
    
    Paramètres
    ----------
    w_remove : dict
        Exemple : {"random": 1, "worst": 1}
    poids_reparation : dict
        Exemple : {"greedy": 1, "best": 1}

    Retour
    ------
    op_destruct : str
    op_repare : str
    """
    # Tables de poids
    w_remove = {}
    for op, data in scores["remove"].items():
        w_remove[op] = data["weight"]

    w_repair = {}
    for op, data in scores["insert"].items():
        w_repair[op] = data["weight"]

    # Tirage par roulette (pondéré)

    #Générer une probabilité pour chaque opérateur de destruction à partir des poids
    total = sum(w_remove.values())
    proba_destruction = {}
    for k, v in w_remove.items():
        proba_destruction[k] = v / total


    # Tirer un opérateur de destruction selon la probabilité calculée
    op_destruct = random.choices(list(proba_destruction.keys()), weights=list(proba_destruction.values()))[0]

    # Générer une probabilité pour chaque opérateur de réparation à partir des poids
    total = sum(w_repair.values())
    proba_reparation = {}
    for k, v in w_repair.items():
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
        routes_copy = []
        i = 0
        while i < len(routes):
            # copie d'une route
            route_originale = routes[i]
            
            route_copie = []
            j = 0
            while j < len(route_originale):
                route_copie.append(route_originale[j])
                j += 1
            
            routes_copy.append(route_copie)
            i += 1

        return routes_copy, []

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


def reparation_greedy(routes_partial, removed, coords, metric="manhattan",
                      demandes=None, capacite=None, contraintes=None, ordre="as_is"):
    """
    Réinsère toutes les villes de `removed` en mode Greedy (Δ minimal successif).
    Chaque client est inséré à la meilleure position (Δ minimal) sur l’ensemble des routes.
    """
    routes_modifiees = []
    for rt in routes_partial:
        routes_modifiees.append(rt[:])

    delta_total = 0.0
    non_inseres = []
    ordre_effectif = list(removed)

    for node in ordre_effectif:
        delta_best = float("inf")
        choix_best = (None, None)  # (r_idx, idx)

        # Recherche du meilleur emplacement faisable pour ce client
        for r_idx, route in enumerate(routes_modifiees):
            for i in range(len(route) - 1):
                if insertion_faisable(route, i, node, demandes, capacite, contraintes):
                    # Calcul du coût d’insertion
                    a = route[i]
                    b = route[i + 1]
                    ax, ay = coords[a]
                    bx, by = coords[b]
                    nx, ny = coords[node]

                    if metric == "manhattan":
                        delta = (abs(ax - nx) + abs(ay - ny)) + (abs(nx - bx) + abs(ny - by)) - (abs(ax - bx) + abs(ay - by))
                    else:
                        delta = (math.hypot(ax - nx, ay - ny) + math.hypot(nx - bx, ny - by) - math.hypot(ax - bx, ay - by))

                    # Meilleur emplacement trouvé ?
                    if delta < delta_best:
                        delta_best = delta
                        choix_best = (r_idx, i)

        # Si aucun emplacement faisable trouvé
        if choix_best == (None, None):
            non_inseres.append(node)
            continue

        # Appliquer l’insertion optimale
        r, i = choix_best
        routes_modifiees[r].insert(i + 1, node)
        if delta_best is None:
            delta_total += 0.0
        else:
            delta_total += delta_best
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

def regret_reparation(routes_partial, removed, coords, metric="manhattan",
                     demandes=None, capacite=None, contraintes=None, regret_level=2):
    """
    Réinsère les villes en utilisant l'heuristique de regret.
    Le regret d'un client est la différence entre son meilleur coût d'insertion
    et son second meilleur coût d'insertion (ou n-ème meilleur selon regret_level).
    On insère en priorité le client avec le plus grand regret.
    """
    routes_modifiees = []
    for rt in routes_partial:
        routes_modifiees.append(rt[:])

    delta_total = 0.0
    non_inseres = []
    remaining = list(removed)
    
    while remaining:
        best_regret = -1
        best_client = None
        best_insertion = None
        
        # Calculer le regret pour chaque client restant
        for client in remaining:
            # Trouver tous les coûts d'insertion possibles pour ce client
            insertion_costs = []
            
            for r_idx, route in enumerate(routes_modifiees):
                for i in range(len(route) - 1):
                    if insertion_faisable(route, i, client, demandes, capacite, contraintes):
                        delta = delta_insertion(route, i, client, coords, metric)
                        insertion_costs.append((delta, r_idx, i))
            
            if len(insertion_costs) == 0:
                # Aucune insertion possible pour ce client
                continue
            
            # Trier par coût croissant
            insertion_costs.sort(key=lambda x: x[0])
            
            # Calculer le regret
            if len(insertion_costs) >= regret_level:
                regret = insertion_costs[regret_level - 1][0] - insertion_costs[0][0]
            else:
                # Si moins de positions que regret_level, utiliser la différence max
                if len(insertion_costs) > 1:
                    regret = insertion_costs[-1][0] - insertion_costs[0][0]
                else:
                    regret = insertion_costs[0][0]

            
            # Garder le client avec le plus grand regret
            if regret > best_regret:
                best_regret = regret
                best_client = client
                best_insertion = insertion_costs[0]  # Meilleure position pour ce client
        
        # Si aucun client ne peut être inséré
        if best_client is None:
            non_inseres.extend(remaining)
            break
        
        # Insérer le client avec le plus grand regret à sa meilleure position
        cost, r_idx, i = best_insertion
        routes_modifiees[r_idx].insert(i + 1, best_client)
        delta_total += cost
        remaining.remove(best_client)
    
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
        T = refroidissement(T, params)

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

    # Créditer opérateurs (score + uses) - remove ET insert
    pi = scores["pi"]
    credit = pi.get(outcome, 0.0)
    scores["remove"][op_remove]["score"] += credit
    scores["remove"][op_remove]["uses"] += 1
    scores["insert"][op_insert]["score"] += credit
    scores["insert"][op_insert]["uses"] += 1

    # Refroidissement éventuel
    T = refroidissement(T, params)

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
        "pi": {  
            "best_global": 50.0,  
            "improve": 25.0,     
            "accepted_worse": 3.0
        },
        "rho": 0.65,
        "segment_len": 25,
        "iters_in_segment": 0
    }

    for op in remove_ops:
        scores["remove"][op] = {"score": 0.0, "uses": 0, "weight": 1.0}
    for op in insert_ops:
        scores["insert"][op] = {"score": 0.0, "uses": 0, "weight": 1.0}

    params = {
        "accept_mode": "sa",  
        "alpha": 0.996,
        "epsilon": 0.0 
    }

    return scores, params

def refroidissement(T, params):
    """
    Refroidissement simple : T <- alpha * T (simulated annealing)
    """
    if params.get("accept_mode", "sa") == "sa":
        T = params.get("alpha", 0.995) * T
    return T

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

def alns(initial_routes, coords,
         metric="manhattan",
         n_iter=500,
         q_remove=2,
         demandes=None,
         capacite=None,
         contraintes=None,
         seed=None,
         log=50,
         max_time=None):
    """
    Lance l'ALNS sur une solution initiale.
    Retourne l'état final (incluant le meilleur global).
    """

    # Démarrage du chronomètre
    start_time = time.perf_counter()

    if seed is not None:
        random.seed(seed)

    # Opérateurs disponibles - ajout de diversité
    remove_ops = ["worst", "random", "shaw"]
    insert_ops = ["greedy", "best", "regret"]

    # Init scores & params
    scores, params = init_scores_et_param(remove_ops, insert_ops)

    # Température initiale élevée mais optimisée pour convergence rapide
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
        "regret": lambda routes_partial, removed, coords, metric, **kw:
            regret_reparation(routes_partial, removed, coords, metric, **kw),
    }

    # Boucle ALNS
    it = 1
    while it <= n_iter:
        # 1 Sélection opérateurs via poids
        op_remove, op_insert = selection_op(scores)

        # 2 Récup fonctions
        remove_func = remove_funcs.get(op_remove)
        repair_func = repair_funcs.get(op_insert)
        if remove_func is None or repair_func is None:
            # Opérateur inconnu -> on passe (sécurité)
            it += 1
            continue

        # Une itération ALNS 
        state, T, scores, outcome = alns_iteration(
            state, coords, metric, T, params, scores,
            op_remove, op_insert, remove_func, repair_func, q_remove,
            demandes=demandes, capacite=capacite, contraintes=contraintes
        )

        # Vérification du temps limite
        if max_time is not None and time.perf_counter() - start_time > max_time:
            print(f"Temps maximum écoulé ({max_time}s). Arrêt de l'ALNS.")
            return state

        # Logs
        if (log is not None) and (it % log == 0):
            print(f"[ALNS] it={it:5d} | destroy={op_remove:7s} | repair={op_insert:7s} | outcome={outcome:15s} | C={state['C']:.2f} | C*={state['C_best']:.2f} | T={T:.4f}")

        it += 1

    # 6) Retour
    return state

def gap(cout, fichier_vrp):
    """
    Calcule l'écart (en %) entre un coût courant et le coût optimal indiqué
    dans le fichier .sol correspondant en utilisant vrplib.read_solution.
    Retourne un float (pourcentage) ou None si le .sol est introuvable ou illisible.
    """
    try:
        # Construire le chemin vers le fichier .sol
        nom_fichier = os.path.basename(fichier_vrp)
        base, _ = os.path.splitext(nom_fichier)
        sol_path = os.path.join("data", base + ".sol")
        
        # Utiliser vrplib pour lire la solution
        solution = vrplib.read_solution(sol_path)
        opt_cost = solution["cost"]
        
        if opt_cost == 0:
            return None
        
        return 100.0 * (cout - opt_cost) / opt_cost
        
    except FileNotFoundError:
        print(f"Fichier .sol non trouvé: {sol_path}")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier .sol avec vrplib: {e}")
        return None

def tracer_vrp(instance, routes=None, titre="Clients et Dépôts"):
    """
    Trace les clients, dépôts et routes en utilisant les données VRPLib.
    """
    coords = instance["node_coord"]  # Numpy array (n, 2)
    depot = instance["depot"][0]     # Index du dépôt
    n = instance["dimension"]        # Nombre de nœuds
    
   # Séparer clients et dépôts
    clients = []
    i = 0
    while i < n:
        if i != depot:
            clients.append(i)
        i += 1

    depots = [depot]

    # Points clients
    x_clients = []
    y_clients = []

    i = 0
    while i < len(clients):
        cid = clients[i]
        x_clients.append(coords[cid][0])
        y_clients.append(coords[cid][1])
        i += 1

    # Points dépôts
    x_depots = []
    y_depots = []

    i = 0
    while i < len(depots):
        did = depots[i]
        x_depots.append(coords[did][0])
        y_depots.append(coords[did][1])
        i += 1

    # Plot
    plt.figure()
    plt.scatter(x_clients, y_clients, c='blue', label='Clients')
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    # Labels
    for i in range(n):
        x, y = coords[i]
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

# Utilisation des données VRPLib
coords = get_coords_dict(instance)
routes = solution_initiale(instance)
tracer_vrp(instance, routes, titre="Solution Initiale VRP")
print("cout total de la solution initiale :", cout_total(routes, coords, metric="manhattan"))

# ALNS
debut = time.perf_counter()
tracemalloc.start()
state_final = alns(
    initial_routes=routes,
    coords=coords,
    metric="euclidienne", 
    n_iter=1500, 
    q_remove=7,
    demandes=get_demands_dict(instance),
    capacite=instance["capacity"],
    contraintes=None,
    seed=42,
    log=100,
    max_time=300 
)

print("Coût final       :", state_final["C"])
print("Meilleur global :", state_final["C_best"])
print("Écart (%)    :", gap(state_final["C_best"], choix_fichier))

fin = time.perf_counter()
print("Temps d'exécution :", fin - debut, "secondes")
current, peak = tracemalloc.get_traced_memory()
print(f"Mémoire actuelle : {current / 1024:.2f} Ko")

tracer_vrp(instance, state_final["S_best"], titre="ALNS: meilleure solution")