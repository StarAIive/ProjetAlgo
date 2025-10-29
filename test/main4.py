import os
import glob
import math
import matplotlib.pyplot as plt

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

def creer_solution_initiale(fichier):
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)
    capacites = lire_capacite(fichier)
    demandes = lire_demandes(fichier)

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
    print("Clients :", clients)

    tour = [depot]
    demande = 0
    p = 0
    while p < len(clients):
        if clients[p] in demandes:
            demande += demandes[clients[p]]
            if capacites is not None and demande > capacites:
                tour.append(depot)
                demande = 0
            else:
                tour.append(clients[p])
                p += 1
        
        tour.append(depot)
    print("Tournée initiale :", tour)

    # Découpage des routes
    routes = []      # liste de routes
    route = []       # route courante

    i = 0
    n = len(tour)
    while i < n:
        if tour[i] == depot:
            # Si la route contient déjà des clients, on la ferme
            if len(route) > 1:
                route.append(depot)
                routes.append(route)
                route = []
            # Toujours redémarrer par un dépôt
            route.append(depot)
        else:
            route.append(tour[i])
        i += 1

    # Dernière vérification : si la route a été commencée mais pas fermée
    if len(route) > 1:
        route.append(depot)
        routes.append(route)

    print("Routes découpées :", routes)
    return tour

def calculer_cout(tour, coords):
    cout = 0.0
    i = 0
    while i < len(tour) - 1:
        u = tour[i]
        v = tour[i + 1]
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        cout += math.hypot(x2 - x1, y2 - y1)
        i += 1
    return cout

def generer_voisins_swap(tour):
    voisins = []
    i = 1
    while i < len(tour) - 1:      # ne pas toucher aux dépôts
        j = i + 1
        while j < len(tour) - 1:
            v = tour[:]           # copie
            tmp = v[i]
            v[i] = v[j]
            v[j] = tmp
            voisins.append(v)
            j += 1
        i += 1
    return voisins

def meilleur_voisin(voisins, coords):
    meilleur = None
    meilleur_cout = float("inf")
    k = 0
    while k < len(voisins):
        v = voisins[k]
        c = calculer_cout(v, coords)
        if c < meilleur_cout:
            meilleur_cout = c
            meilleur = v
        k += 1
    return meilleur, meilleur_cout

def amelioration_locale(tour, coords):
    amelioration = True
    cout_actuel = calculer_cout(tour, coords)
    while amelioration:
        voisins = generer_voisins_swap(tour)
        meilleur, meilleur_cout = meilleur_voisin(voisins, coords)
        if meilleur_cout < cout_actuel:
            tour = meilleur
            cout_actuel = meilleur_cout
        else:
            amelioration = False
    return tour, cout_actuel

def tabu_search(tour_initiale, coords, iterations=50, tenure=5):
    S = tour_initiale[:]
    S_cost = calculer_cout(S, coords)
    best = S[:]
    best_cost = S_cost
    tabu = {} 

    k = 1
    while k <= iterations:
        best_cand = None
        best_cand_cost = float("inf")
        best_cand_move = None

        n = len(S)
        i = 1
        while i < n - 1:
            j = i + 1
            while j < n - 1:
                cand = S[:]
                tmp = cand[i]
                cand[i] = cand[j]
                cand[j] = tmp

                c = calculer_cout(cand, coords)
                move = (i, j)

                est_tabu = (move in tabu and tabu[move] >= k)
                aspiration = (c < best_cost)

                if (not est_tabu) or aspiration:
                    if c < best_cand_cost:
                        best_cand = cand
                        best_cand_cost = c
                        best_cand_move = move
                j += 1
            i += 1

        if best_cand is None:
            break

        S = best_cand
        S_cost = best_cand_cost
        tabu[best_cand_move] = k + tenure

        if S_cost < best_cost:
            best = S[:]
            best_cost = S_cost

        k += 1

    return best, best_cost

def tracer_vrp(fichier, tour=None, titre="Clients et Dépôts"):
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
        x_clients.append(x)
        y_clients.append(y)

    x_depots, y_depots = [], []
    for did in depots:
        x, y = coords[did]
        x_depots.append(x)
        y_depots.append(y)

    # Plot
    plt.figure()
    plt.scatter(x_clients, y_clients, c='blue', label='Clients')
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    # Labels
    for i, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, color="black")

    # Route si fournie
    if tour is not None and len(tour) >= 2:
        i = 0
        while i < len(tour) - 1:
            x1, y1 = coords[tour[i]]
            x2, y2 = coords[tour[i + 1]]
            plt.plot([x1, x2], [y1, y2], 'k-')
            i += 1

    plt.title(titre)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

tour = creer_solution_initiale(choix_fichier)
coords = lire_coordonnees(choix_fichier)

print("Capacité :", lire_capacite(choix_fichier))
print("Coût de la tournée initiale :", calculer_cout(tour, coords))

# Affichage initial
tracer_vrp(choix_fichier, tour, titre="Tournée initiale")

# Tabu Search
best_tabu, best_tabu_cost = tabu_search(tour, coords, iterations=100, tenure=7)
print("Coût de la meilleure tournée (Tabu) :", best_tabu_cost)

# Affichage résultat
tracer_vrp(choix_fichier, best_tabu, titre="Tournée améliorée (Tabu)")
