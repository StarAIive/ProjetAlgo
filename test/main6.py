import os
import glob
import math
import matplotlib.pyplot as plt
import random

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

def matrice_distances(coords, metric="euclidienne"):
    ids = sorted(list(coords.keys()))
    idx = {}
    i = 0
    while i < len(ids):
        node_id = ids[i]
        idx[node_id] = i
        i += 1

    n = len(ids)
    M = []
    i = 0
    while i < n:
        ligne = []
        j = 0
        while j < n:
            ligne.append(0.0)
            j += 1
        M.append(ligne)
        i += 1

    for i in range(n):
        xa, ya = coords[ids[i]]
        for j in range(n):
            xb, yb = coords[ids[j]]
            if metric == "euclidienne":
                dist = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            else:  # "manhattan"
                dist = abs(xa - xb) + abs(ya - yb)
            M[i][j] = dist

    return ids, idx, M

def cout_route(route, M, idx):
    total = 0.0
    i = 0
    while i < len(route) - 1:
        a = route[i]
        b = route[i + 1]
        ia = idx[a]
        ib = idx[b]
        total += M[ia][ib]
        i += 1
    return total


def cout_solution(routes, M, idx):
    total = 0.0
    for r in routes:
        c_r = cout_route(r, M, idx)
        total += c_r
    return total

def delta_insertion(route, pos, client_id, M, idx):
    if pos < 1 or pos >= len(route):
        raise ValueError("Position d'insertion invalide.")

    u = route[pos - 1]
    v = route[pos]
    iu = idx[u]
    iv = idx[v]
    ic = idx[client_id]
    cout_avant = M[iu][iv]
    cout_apres = M[iu][ic] + M[ic][iv]
    delta = cout_apres - cout_avant
    return delta

def inserer_client(route, pos, client_id):
    new_route = route[:]
    new_route.insert(pos, client_id)
    return new_route

def choisir_client(non_servis, demandes, depot_id=None, coords=None, idx=None, M=None):
    c_best = None
    best = -1
    for c in non_servis:
        d = demandes.get(c, 0)
        if d > best:
            best = d; c_best = c
    return c_best


def grasp(coords, demandes, capacite, depot_id, M, idx,
                    alpha=0.3, q_routes=8, k_pos=10):
    routes = [[depot_id, depot_id]]
    non_servis = set(demandes.keys())

    while non_servis:
        # boucle principale tant que non_servis non vide
        c = choisir_client(non_servis, demandes, depot_id, coords, idx, M)

        # générer les candidats d'insertion faisables (capacité respectée)
        candidats = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                # Vérification de capacité (charge de la route + demande du client)
                charge_route = 0
                for client in route:
                    if client in demandes:
                        charge_route += demandes[client]
                if charge_route + demandes[c] <= capacite:
                    delta = delta_insertion(route, pos, c, M, idx)
                    candidats.append((c, r_idx, pos, delta))

        # construire la RCL (seuil alpha ou taille fixe)
        if candidats:
            delta_min = min(c[3] for c in candidats)
            delta_max = max(c[3] for c in candidats)
            rcl = [c for c in candidats if c[3] <= delta_min + alpha * (delta_max - delta_min)]
        else:
            rcl = []

        # choisir un candidat au hasard dans la RCL et l'appliquer
        if rcl:
            c, r_idx, pos, delta = random.choice(rcl)
            routes[r_idx] = inserer_client(routes[r_idx], pos, c)
            non_servis.remove(c)
        else:
            # si aucun candidat -> ouvrir une nouvelle route et réessayer
            if demandes[c] <= capacite:
                routes.append([depot_id, depot_id])
                continue
            else:
                raise ValueError("Demande trop élevée pour un nouveau camion.")

    return routes




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


routes = solution_initiale(choix_fichier)
tracer_vrp(choix_fichier, routes, titre="Solution Initiale VRP")

coords = lire_coordonnees(choix_fichier)
ids, idx, M = matrice_distances(coords, metric="euclidienne")
cout_init = cout_solution(routes, M, idx)
print("Coût initial de la solution :", cout_init)

demandes = lire_demandes(choix_fichier)
capacite = lire_capacite(choix_fichier)
depot_id = lire_depots(choix_fichier)[0]

best_routes = grasp(coords, demandes, capacite, depot_id, M, idx)
best_cost = cout_solution(best_routes, M, idx)

print("Coût après GRASP :", best_cost)

# Visualisation résultat optimisé
tracer_vrp(choix_fichier, best_routes, titre="Solution Optimisée VRP (GRASP)")

