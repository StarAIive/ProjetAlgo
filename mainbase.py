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

def cout_total(routes, coords, metric="euclidienne"):
    total = 0.0
    for route in routes:
        if not route or len(route) < 2:
            continuea
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
print("cout total de la solution initiale :", cout_total(routes, lire_coordonnees(choix_fichier)))


