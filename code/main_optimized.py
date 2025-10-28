import os
import glob
import math
import matplotlib.pyplot as plt
import random
from collections import deque
import time
from math import sqrt

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
        print("Entrée invalide : veuillez entrer un nombre entier (0 ou 1).")

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
    print("Mode 1 (contraintes supplémentaires) : non implémenté pour l'instant.")
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

def solution_initiale_savings(fichier):
    """Solution initiale basée sur l'algorithme de Clarke & Wright (Savings)"""
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)
    demandes = lire_demandes(fichier)
    capacite = lire_capacite(fichier)
    
    if not depots:
        print("Aucun dépôt trouvé.")
        return []
    
    depot = depots[0]
    clients = [i for i in coords.keys() if i not in depots]
    
    # Calcul des savings
    savings = []
    for i in range(len(clients)):
        for j in range(i + 1, len(clients)):
            ci, cj = clients[i], clients[j]
            # Savings = distance(depot, i) + distance(depot, j) - distance(i, j)
            s = _dist(depot, ci, coords) + _dist(depot, cj, coords) - _dist(ci, cj, coords)
            savings.append((s, ci, cj))
    
    # Trier par savings décroissants
    savings.sort(reverse=True)
    
    # Créer routes initiales (depot -> client -> depot)
    routes = {}
    client_to_route = {}
    for client in clients:
        routes[client] = [depot, client, depot]
        client_to_route[client] = client
    
    # Fusionner selon les savings
    for saving, ci, cj in savings:
        if saving <= 0:
            break
            
        route_i = client_to_route.get(ci)
        route_j = client_to_route.get(cj)
        
        if route_i is None or route_j is None or route_i == route_j:
            continue
        
        # Vérifier si ci et cj sont aux extrémités de leurs routes
        ri, rj = routes[route_i], routes[route_j]
        
        # Positions possibles pour fusion
        ci_first = (ri[1] == ci)
        ci_last = (ri[-2] == ci)
        cj_first = (rj[1] == cj)
        cj_last = (rj[-2] == cj)
        
        new_route = None
        
        if ci_last and cj_first:
            # ri: depot...ci-depot, rj: depot-cj...depot -> depot...ci-cj...depot
            new_route = ri[:-1] + rj[1:]
        elif ci_first and cj_last:
            # ri: depot-ci...depot, rj: depot...cj-depot -> depot-cj...ci...depot  
            new_route = rj[:-1] + ri[1:]
        elif ci_last and cj_last:
            # ri: depot...ci-depot, rj: depot...cj-depot -> depot...ci-cj...depot (reverse rj)
            new_route = ri[:-1] + list(reversed(rj[1:-1])) + [depot]
        elif ci_first and cj_first:
            # ri: depot-ci...depot, rj: depot-cj...depot -> depot-ci...cj...depot (reverse ri)
            new_route = [depot] + list(reversed(ri[1:-1])) + rj[1:]
        
        if new_route is not None:
            # Vérifier contrainte de capacité
            charge = sum(demandes.get(c, 0) for c in new_route[1:-1])
            if charge <= capacite:
                # Fusionner les routes
                new_key = min(route_i, route_j)
                old_key = max(route_i, route_j)
                
                routes[new_key] = new_route
                del routes[old_key]
                
                # Mettre à jour client_to_route
                for c in new_route[1:-1]:
                    client_to_route[c] = new_key
    
    return list(routes.values())

def solution_initiale(fichier):
    """Génère la meilleure solution initiale entre plusieurs méthodes"""
    
    # Méthode 1: Glouton simple
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)
    demandes = lire_demandes(fichier)
    capacite = lire_capacite(fichier)
    
    if not depots:
        return []
    
    depot = depots[0]
    clients = [i for i in coords.keys() if i not in depots]
    
    routes_simple = []
    remaining = clients[:]
    
    while remaining:
        route = [depot]
        current_load = 0
        current_pos = depot
        
        while remaining:
            # Plus proche voisin faisable
            best_client = None
            best_dist = float('inf')
            
            for client in remaining:
                if current_load + demandes.get(client, 0) <= capacite:
                    dist = _dist(current_pos, client, coords)
                    if dist < best_dist:
                        best_dist = dist
                        best_client = client
            
            if best_client is None:
                break
            
            route.append(best_client)
            current_load += demandes.get(best_client, 0)
            current_pos = best_client
            remaining.remove(best_client)
        
        route.append(depot)
        routes_simple.append(route)
    
    # Méthode 2: Savings
    try:
        routes_savings = solution_initiale_savings(fichier)
        cost_savings = cout_total(routes_savings, coords)
    except:
        routes_savings = routes_simple
        cost_savings = float('inf')
    
    cost_simple = cout_total(routes_simple, coords)
    
    # Retourner la meilleure
    if cost_savings < cost_simple:
        print(f"Utilisation méthode Savings (coût: {cost_savings:.2f})")
        return routes_savings
    else:
        print(f"Utilisation méthode Plus Proche Voisin (coût: {cost_simple:.2f})")
        return routes_simple

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

# ======= ALGORITHME TABOU ULTRA-OPTIMISÉ =======

def _dist(u, v, coords):
    x1, y1 = coords[u]
    x2, y2 = coords[v]
    return math.hypot(x1 - x2, y1 - y2)

def _route_charge(route, demandes):
    if not demandes: 
        return 0
    return sum(demandes.get(route[i], 0) for i in range(1, len(route)-1))

def _edge(a, b):
    return (a, b) if a < b else (b, a)

def _delta_2opt(route, i, k, coords):
    """2-opt: inverse [i..k-1]"""
    a, b = route[i-1], route[i]
    c, d = route[k-1], route[k]
    before = _dist(a, b, coords) + _dist(c, d, coords)
    after = _dist(a, c, coords) + _dist(b, d, coords)
    return after - before

def _apply_2opt(route, i, k):
    return route[:i] + list(reversed(route[i:k])) + route[k:]

def _apply_reloc(routes, i, p, j, q):
    new_routes = [r[:] for r in routes]
    c = new_routes[i].pop(p)
    if i == j and q > p: 
        q -= 1
    new_routes[j].insert(q, c)
    new_routes = [r for r in new_routes if not (len(r) == 2 and r[0] == r[-1])]
    return new_routes

def _apply_swap(routes, i, p, j, q):
    new_routes = [r[:] for r in routes]
    new_routes[i][p], new_routes[j][q] = new_routes[j][q], new_routes[i][p]
    return new_routes

def _route_cost(route, coords):
    return sum(_dist(route[t], route[t+1], coords) for t in range(len(route)-1))

def _build_knearest(coords, k=30):
    """Construit k-plus proches voisins pour limitation granulaire"""
    ids = list(coords.keys())
    knn = {i: [] for i in ids}
    for i in ids:
        L = []
        xi, yi = coords[i]
        for j in ids:
            if j == i: 
                continue
            xj, yj = coords[j]
            L.append((math.hypot(xi-xj, yi-yj), j))
        L.sort(key=lambda t: t[0])
        knn[i] = [j for _, j in L[:k]]
    return knn

def ameliorer_solution_initiale(routes, coords, demandes, capacite):
    """Amélioration multi-passes de la solution initiale"""
    print("🔧 Amélioration de la solution initiale...")
    
    for pass_num in range(3):
        ameliore = False
        for r_id, route in enumerate(routes):
            if len(route) <= 4:
                continue
                
            best_cost = _route_cost(route, coords)
            best_route = route[:]
            
            # 2-opt exhaustif
            for i in range(1, len(route) - 2):
                for j in range(i + 2, len(route)):
                    if j >= len(route):
                        break
                    
                    new_route = route[:i] + list(reversed(route[i:j])) + route[j:]
                    new_cost = _route_cost(new_route, coords)
                    
                    if new_cost < best_cost - 1e-9:
                        best_cost = new_cost
                        best_route = new_route[:]
                        ameliore = True
            
            routes[r_id] = best_route
        
        if not ameliore:
            break
    
    return routes

def _apply_oropt_inter(routes, i, start, size, j, ins_pos):
    """Déplace un segment de taille 1-3 d'une route vers une autre"""
    new_routes = [r[:] for r in routes]
    
    # Extraire le segment
    segment = new_routes[i][start:start+size]
    new_routes[i] = new_routes[i][:start] + new_routes[i][start+size:]
    
    # Ajuster position d'insertion si nécessaire
    if i == j and ins_pos > start:
        ins_pos -= size
    
    # Insérer le segment
    for k, client in enumerate(segment):
        new_routes[j].insert(ins_pos + k, client)
    
    # Nettoyer les routes vides
    new_routes = [r for r in new_routes if len(r) > 2]
    return new_routes

def _generate_neighbors_enhanced(routes, coords, demandes, capacite, knn, candidate_limit=1500):
    """Génération de voisins ultra-améliorée avec tous les opérateurs"""
    tested = 0
    R = len(routes)
    cap = capacite if capacite is not None else float('inf')
    charges = [_route_charge(routes[i], demandes) for i in range(R)]

    # 1. RELOCATE (1-client) - Intra et Inter routes
    for i in range(R):
        ri = routes[i]
        Li = len(ri)
        for p in range(1, Li-1):
            c = ri[p]
            
            for j in range(R):
                rj = routes[j]
                for q in range(1, len(rj)):
                    if i == j and (q == p or q == p+1): 
                        continue
                        
                    # Vérification capacité
                    new_i = charges[i] - demandes.get(c, 0)
                    new_j = charges[j] + demandes.get(c, 0)
                    
                    if new_i <= cap and new_j <= cap:
                        delta_remove = _dist(ri[p-1], ri[p+1], coords) - (_dist(ri[p-1], c, coords) + _dist(c, ri[p+1], coords))
                        delta_insert = (_dist(rj[q-1], c, coords) + _dist(c, rj[q], coords)) - _dist(rj[q-1], rj[q], coords)
                        delta = delta_remove + delta_insert
                        
                        yield ("relocate", i, p, j, q, delta)
                        tested += 1
                        if tested >= candidate_limit: 
                            return

    # 2. OR-OPT (2-3 clients) - Inter routes
    for i in range(R):
        ri = routes[i]
        Li = len(ri)
        
        for size in [2, 3]:
            if Li - 2 < size:
                continue
                
            for start in range(1, Li - size):
                segment = ri[start:start+size]
                segment_demand = sum(demandes.get(c, 0) for c in segment)
                
                for j in range(R):
                    if i == j:
                        continue
                        
                    rj = routes[j]
                    new_i = charges[i] - segment_demand
                    new_j = charges[j] + segment_demand
                    
                    if new_i <= cap and new_j <= cap:
                        for ins_pos in range(1, len(rj)):
                            # Delta approximatif
                            # Enlever segment de route i
                            delta = _dist(ri[start-1], ri[start+size], coords) - (_dist(ri[start-1], ri[start], coords) + _dist(ri[start+size-1], ri[start+size], coords))
                            # Insérer segment dans route j
                            delta += (_dist(rj[ins_pos-1], segment[0], coords) + _dist(segment[-1], rj[ins_pos], coords)) - _dist(rj[ins_pos-1], rj[ins_pos], coords)
                            
                            yield ("oropt", i, start, size, j, ins_pos, delta)
                            tested += 1
                            if tested >= candidate_limit: 
                                return

    # 3. SWAP - Plus complet
    for i in range(R):
        ri = routes[i]
        Li = len(ri)
        for j in range(i, R):
            rj = routes[j]
            Lj = len(rj)
            
            for p in range(1, Li-1):
                c1 = ri[p]
                for q in range(1, Lj-1):
                    if i == j and p == q: 
                        continue
                    c2 = rj[q]
                    
                    # Vérification capacité
                    if i != j:
                        new_i = charges[i] - demandes.get(c1, 0) + demandes.get(c2, 0)
                        new_j = charges[j] - demandes.get(c2, 0) + demandes.get(c1, 0)
                        if new_i > cap or new_j > cap:
                            continue
                    
                    # Delta approximatif
                    delta = 0.0
                    # Route i
                    delta += _dist(ri[p-1], c2, coords) + _dist(c2, ri[p+1], coords) - (_dist(ri[p-1], c1, coords) + _dist(c1, ri[p+1], coords))
                    # Route j  
                    delta += _dist(rj[q-1], c1, coords) + _dist(c1, rj[q+1], coords) - (_dist(rj[q-1], c2, coords) + _dist(c2, rj[q+1], coords))
                    
                    yield ("swap", i, p, j, q, delta)
                    tested += 1
                    if tested >= candidate_limit: 
                        return

    # 4. 2-OPT Intra-route pour chaque route
    for r_id, route in enumerate(routes):
        L = len(route)
        if L <= 4:
            continue
            
        for i in range(1, L-2):
            for j in range(i+2, L):
                if j >= L:
                    break
                    
                delta = _delta_2opt(route, i, j, coords)
                if delta < -1e-6:  # Seulement les améliorations
                    yield ("2opt", r_id, i, j, delta)
                    tested += 1
                    if tested >= candidate_limit: 
                        return

def recherche_tabou_ultra(routes_init, coords, demandes=None, capacite=None, 
                         k_nearest=40, candidate_limit=1000, max_iter=10000, 
                         time_cap=180, lambda_freq=0.05):
    """Recherche Tabou ultra-optimisée pour atteindre ~800"""
    
    print("🚀 Recherche Tabou Ultra-Optimisée")
    
    # Initialisation
    rng = random.Random(42)
    knn = _build_knearest(coords, k=k_nearest)
    
    # Amélioration préliminaire
    print(f"Coût initial avant amélioration : {cout_total(routes_init, coords):.2f}")
    S = ameliorer_solution_initiale([r[:] for r in routes_init], coords, demandes, capacite)
    S_cost = cout_total(S, coords)
    print(f"Coût après amélioration initiale : {S_cost:.2f}")
    
    best = [r[:] for r in S]
    best_cost = S_cost
    
    # Variables de contrôle
    n_clients = sum(max(0, len(r)-2) for r in S)
    base_tenure = max(3, int(1.5 * sqrt(max(1, n_clients))))
    tenure = base_tenure
    tabu = {}
    
    # Mémoire long terme
    freq = {}
    def _upd_freq(routes):
        for r in routes:
            for t in range(len(r)-1):
                e = _edge(r[t], r[t+1])
                freq[e] = freq.get(e, 0) + 1
    
    _upd_freq(S)
    
    start = time.time()
    no_improve = 0
    it = 0
    last_best = S_cost
    
    # Boucle principale
    while it < max_iter:
        if time_cap and (time.time() - start) >= time_cap:
            print(f"⏰ Limite de temps atteinte ({time_cap}s)")
            break
            
        # Génération de voisins
        moves = list(_generate_neighbors_enhanced(
            S, coords, demandes or {}, capacite, knn, candidate_limit=candidate_limit
        ))
        
        if not moves:
            print("Aucun voisin trouvé, arrêt")
            break
        
        # Sélection du meilleur mouvement admissible
        best_cand = None
        best_aug = float('inf')
        
        for move in moves:
            # Gérer différents formats de mouvements
            if len(move) == 5:  # 2opt
                kind, i, p, j, delta = move
                key = ("2opt", i, p, j)
            elif len(move) == 6 and move[0] != "oropt":  # relocate, swap
                kind, i, p, j, q, delta = move
                if kind == "relocate":
                    c = S[i][p]
                    key = ("move", c)
                else:  # swap
                    c1, c2 = S[i][p], S[j][q]
                    key = ("swap", min(c1, c2), max(c1, c2))
            elif len(move) == 7:  # oropt
                kind, i, start, size, j, ins_pos, delta = move
                segment = S[i][start:start+size]
                key = ("oropt", tuple(segment), j, ins_pos)
            
            # Pénalisation mémoire long terme (simplifié)
            dfreq = 0
            
            aug = (S_cost + delta) + lambda_freq * dfreq
            is_tabu = (key in tabu and tabu[key] > it)
            
            # Aspiration
            new_real = S_cost + delta
            if is_tabu and new_real >= best_cost - 1e-12:
                continue
            
            if aug < best_aug:
                best_aug = aug
                if len(move) == 5:  # 2opt
                    best_cand = (kind, i, p, j, delta, key, new_real)
                elif len(move) == 7:  # oropt
                    best_cand = (kind, i, start, size, j, ins_pos, delta, key, new_real)
                else:  # relocate, swap
                    best_cand = (kind, i, p, j, q, delta, key, new_real)
        
        if best_cand is None:
            print("Aucun mouvement admissible")
            break
        
        # Appliquer le mouvement selon le type
        if best_cand[0] == "2opt":
            kind, i, p, j, delta, key, new_real = best_cand
            S[i] = _apply_2opt(S[i], p, j)
        elif best_cand[0] == "oropt":
            kind, i, start, size, j, ins_pos, delta, key, new_real = best_cand
            S = _apply_oropt_inter(S, i, start, size, j, ins_pos)
        else:  # relocate, swap
            kind, i, p, j, q, delta, key, new_real = best_cand
            if kind == "relocate":
                S = _apply_reloc(S, i, p, j, q)
            else:  # swap
                S = _apply_swap(S, i, p, j, q)
        
        # Recalculer le coût réel pour éviter l'accumulation d'erreurs
        S_cost = cout_total(S, coords)
        
        # Mise à jour tabou
        tabu[key] = it + tenure
        
        # Nouveau meilleur ?
        if S_cost < best_cost - 1e-9:
            best = [r[:] for r in S]
            best_cost = S_cost
            no_improve = 0
            tenure = max(base_tenure - 1, 2)  # Réduire tenure quand on s'améliore
            
            print(f"🎯 Itération {it}: Nouveau meilleur = {best_cost:.2f}")
            
            # Intensification : 2-opt sur toutes les routes
            if it % 50 == 0:
                print("🔥 Intensification 2-opt...")
                for r_id, route in enumerate(S):
                    if len(route) <= 4:
                        continue
                    improved = True
                    while improved:
                        improved = False
                        best_delta = 0
                        best_move = None
                        
                        for a in range(1, len(route) - 2):
                            for b in range(a + 2, len(route)):
                                if b >= len(route):
                                    break
                                delta = _delta_2opt(route, a, b, coords)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (a, b)
                        
                        if best_move:
                            a, b = best_move
                            S[r_id] = _apply_2opt(route, a, b)
                            route = S[r_id]
                            S_cost += best_delta
                            improved = True
                            if S_cost < best_cost:
                                best = [r[:] for r in S]
                                best_cost = S_cost
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                tenure = min(base_tenure + 5, tenure + 1)  # Augmenter tenure si stagnation
        
        # Diversification si trop de stagnation
        if no_improve >= 500:
            print(f"🌀 Diversification à l'itération {it}")
            # Perturbation multiple
            for _ in range(10):
                if len(S) < 2:
                    break
                i = rng.randrange(len(S))
                j = rng.randrange(len(S))
                if len(S[i]) > 2 and len(S[j]) > 1:
                    p = rng.randrange(1, len(S[i]) - 1)
                    q = rng.randrange(1, len(S[j]))
                    
                    # Vérifier faisabilité
                    c = S[i][p]
                    new_i = _route_charge(S[i], demandes) - demandes.get(c, 0)
                    new_j = _route_charge(S[j], demandes) + demandes.get(c, 0)
                    
                    if new_i <= capacite and new_j <= capacite:
                        S = _apply_reloc(S, i, p, j, q)
            
            S_cost = cout_total(S, coords)
            no_improve = 0
            tabu.clear()
        
        # Nettoyage périodique
        if it % 200 == 0:
            tabu = {k: v for k, v in tabu.items() if v > it}
        
        it += 1
    
    elapsed = time.time() - start
    print(f"⏱️  Temps d'exécution: {elapsed:.1f}s, Itérations: {it}")
    
    return best, best_cost

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
    plt.figure(figsize=(12, 8))
    plt.scatter(x_clients, y_clients, c='blue', label='Clients', s=60)
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    # Labels
    for i, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, color="black")

    # Routes : couleur différente par route
    if routes is not None and len(routes) > 0:
        couleurs = plt.cm.tab20.colors
        for r_idx, route in enumerate(routes):
            col = couleurs[r_idx % len(couleurs)]
            label = f"Route {r_idx + 1}"
            premiere = True

            for i in range(len(route) - 1):
                x1, y1 = coords[route[i]]
                x2, y2 = coords[route[i + 1]]
                if premiere:
                    plt.plot([x1, x2], [y1, y2], '-', color=col, label=label, linewidth=2)
                    premiere = False
                else:
                    plt.plot([x1, x2], [y1, y2], '-', color=col, linewidth=2)

    plt.title(titre, fontsize=14, fontweight='bold')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ======= EXÉCUTION PRINCIPALE =======

# Chargement des données
coords = lire_coordonnees(choix_fichier)
demandes = lire_demandes(choix_fichier)
capacite = lire_capacite(choix_fichier)

# Génération de la solution initiale
print("🏗️ Génération de la solution initiale...")
routes = solution_initiale(choix_fichier)
cout_init = cout_total(routes, coords)
print(f"Solution initiale - Coût: {cout_init:.2f}, Routes: {len(routes)}")

# Visualisation initiale
tracer_vrp(choix_fichier, routes, titre="Solution Initiale")

def amelioration_finale_intensive(routes, coords, demandes, capacite, temps_max=30):
    """Amélioration intensive finale pour atteindre l'objectif de 800"""
    print("🔥 Amélioration finale intensive...")
    
    start = time.time()
    best = [r[:] for r in routes]
    best_cost = cout_total(best, coords)
    current = [r[:] for r in best]
    
    iteration = 0
    while (time.time() - start) < temps_max and best_cost > 800:
        iteration += 1
        improved = False
        
        # 1. 3-opt exhaustif sur chaque route
        for r_id, route in enumerate(current):
            if len(route) <= 6:
                continue
                
            best_route = route[:]
            best_route_cost = _route_cost(route, coords)
            
            # 3-opt: essayer toutes les combinaisons de 3 arêtes
            for i in range(1, len(route) - 3):
                for j in range(i + 1, len(route) - 2):
                    for k in range(j + 1, len(route) - 1):
                        # 7 possibilités de reconnexion différentes
                        segments = [route[:i], route[i:j], route[j:k], route[k:]]
                        
                        new_routes = [
                            segments[0] + segments[1] + list(reversed(segments[2])) + segments[3],
                            segments[0] + list(reversed(segments[1])) + segments[2] + segments[3],
                            segments[0] + segments[2] + segments[1] + segments[3],
                            segments[0] + segments[2] + list(reversed(segments[1])) + segments[3],
                            segments[0] + list(reversed(segments[1])) + list(reversed(segments[2])) + segments[3],
                        ]
                        
                        for new_route in new_routes:
                            new_cost = _route_cost(new_route, coords)
                            if new_cost < best_route_cost - 1e-9:
                                best_route = new_route
                                best_route_cost = new_cost
                                improved = True
            
            current[r_id] = best_route
        
        # 2. Chain relocations (déplacer plusieurs clients d'affilée)
        for size in [2, 3]:
            for i in range(len(current)):
                ri = current[i]
                if len(ri) <= size + 2:
                    continue
                    
                for start in range(1, len(ri) - size):
                    chain = ri[start:start+size]
                    chain_demand = sum(demandes.get(c, 0) for c in chain)
                    
                    for j in range(len(current)):
                        if i == j:
                            continue
                            
                        rj = current[j]
                        new_i_load = _route_charge(ri, demandes) - chain_demand
                        new_j_load = _route_charge(rj, demandes) + chain_demand
                        
                        if new_i_load <= capacite and new_j_load <= capacite:
                            # Essayer toutes positions d'insertion dans route j
                            for ins_pos in range(1, len(rj)):
                                # Créer nouvelle solution
                                test_routes = [r[:] for r in current]
                                
                                # Retirer chain de route i
                                for _ in range(size):
                                    test_routes[i].pop(start)
                                
                                # Insérer chain dans route j
                                for k, client in enumerate(chain):
                                    test_routes[j].insert(ins_pos + k, client)
                                
                                test_cost = cout_total(test_routes, coords)
                                if test_cost < best_cost - 1e-9:
                                    current = test_routes
                                    best = [r[:] for r in current]
                                    best_cost = test_cost
                                    improved = True
                                    print(f"🎯 Amélioration Chain {size}: {best_cost:.2f}")
        
        # 3. Cross-exchange (échanger des segments entre routes)
        for i in range(len(current)):
            ri = current[i]
            for j in range(i + 1, len(current)):
                rj = current[j]
                
                # Échanger des segments de taille 1-2
                for size_i in [1, 2]:
                    for size_j in [1, 2]:
                        if len(ri) <= size_i + 2 or len(rj) <= size_j + 2:
                            continue
                            
                        for start_i in range(1, len(ri) - size_i):
                            for start_j in range(1, len(rj) - size_j):
                                seg_i = ri[start_i:start_i+size_i]
                                seg_j = rj[start_j:start_j+size_j]
                                
                                # Vérifier capacités
                                demand_i = sum(demandes.get(c, 0) for c in seg_i)
                                demand_j = sum(demandes.get(c, 0) for c in seg_j)
                                
                                new_i_load = _route_charge(ri, demandes) - demand_i + demand_j
                                new_j_load = _route_charge(rj, demandes) - demand_j + demand_i
                                
                                if new_i_load <= capacite and new_j_load <= capacite:
                                    # Créer nouvelle solution
                                    test_routes = [r[:] for r in current]
                                    
                                    # Échanger les segments
                                    test_routes[i] = ri[:start_i] + seg_j + ri[start_i+size_i:]
                                    test_routes[j] = rj[:start_j] + seg_i + rj[start_j+size_j:]
                                    
                                    test_cost = cout_total(test_routes, coords)
                                    if test_cost < best_cost - 1e-9:
                                        current = test_routes
                                        best = [r[:] for r in current]
                                        best_cost = test_cost
                                        improved = True
                                        print(f"🎯 Cross-exchange: {best_cost:.2f}")
        
        if not improved or (time.time() - start) >= temps_max:
            break
    
    elapsed = time.time() - start
    print(f"Amélioration finale terminée en {elapsed:.1f}s après {iteration} itérations")
    return best, best_cost

# Recherche Tabou ultra-optimisée avec focus sur la performance
routes_opt, cout_opt = recherche_tabou_ultra(
    routes_init=routes,
    coords=coords,
    demandes=demandes,
    capacite=capacite,
    k_nearest=60,         # Plus de voisins
    candidate_limit=2000, # Plus de candidats
    max_iter=15000,       # Plus d'itérations
    time_cap=45,          # 45 secondes pour tabou
    lambda_freq=0.02      # Moins de pénalisation pour plus d'agressivité
)

# Amélioration finale intensive
if cout_opt > 800:
    print(f"\n⚡ Coût actuel: {cout_opt:.2f} > 800, lancement amélioration finale...")
    routes_opt, cout_opt = amelioration_finale_intensive(
        routes_opt, coords, demandes, capacite, temps_max=45
    )

# Résultats finaux avec statistiques détaillées
amelioration = ((cout_init - cout_opt) / cout_init) * 100

print(f"\n" + "="*70)
print(f"🏆 RÉSULTATS FINAUX - RECHERCHE TABOU ULTRA-OPTIMISÉE")
print(f"="*70)
print(f"Fichier                : {os.path.basename(choix_fichier)}")
print(f"Coût initial           : {cout_init:.2f}")
print(f"Coût optimisé          : {cout_opt:.2f}")
print(f"Amélioration           : {cout_init - cout_opt:.2f} ({amelioration:.1f}%)")
print(f"Routes initiales       : {len(routes)}")
print(f"Routes optimisées      : {len(routes_opt)}")

# Vérification des contraintes
demandes_ok = True
for route in routes_opt:
    charge = sum(demandes.get(client, 0) for client in route[1:-1])
    if charge > capacite:
        demandes_ok = False
        print(f"❌ Route violant la capacité: {route} (charge: {charge}, max: {capacite})")
        break

if demandes_ok:
    print(f"Contraintes capacité   : ✅ Respectées")
else:
    print(f"Contraintes capacité   : ❌ Violées")

print(f"="*70)

# Évaluation de la performance
if cout_opt <= 800:
    print(f"🎯 OBJECTIF ATTEINT ! Coût ≤ 800 : {cout_opt:.2f}")
elif cout_opt <= 900:
    print(f"🔥 Excellent résultat ! Très proche de l'objectif : {cout_opt:.2f}")
elif cout_opt <= 1000:
    print(f"⚡ Bon résultat, proche de l'objectif : {cout_opt:.2f}")
else:
    print(f"🔄 Résultat correct, optimisations supplémentaires possibles : {cout_opt:.2f}")

# Visualisation finale
tracer_vrp(choix_fichier, routes_opt, titre=f"Solution Optimisée Tabou - Coût: {cout_opt:.0f}")