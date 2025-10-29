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
    if not depots:
        print("Aucun dépôt trouvé.")
        return []
    depot = depots[0]

    clients = []
    for ident in coords.keys():
        est_depot = False
        for d in depots:
            if ident == d:
                est_depot = True
                break
        if not est_depot:
            clients.append(ident)

    tour = [depot]
    p = 0
    while p < len(clients):
        tour.append(clients[p])
        p += 1
    tour.append(depot)
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
    for ident in coords.keys():
        est_depot = False
        for d in depots:
            if ident == d:
                est_depot = True
                break
        if not est_depot:
            clients.append(ident)

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
    for ident, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(ident), fontsize=8, color="black")

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

def split_par_capacite(tour, demandes, capacite, depot_id):
    routes = []           # liste de routes complètes: [dep, ..., dep]
    route_courante = []   # on stocke ici seulement les clients
    charge = 0

    # TODO 1 : parcourir les clients de la tournée (ignorer 1er et dernier qui sont le dépôt)
    i = 1
    while i < len(tour) - 1:
        client = tour[i]
        dem = demandes.get(client, 0)

        # TODO 2 : si on peut charger ce client dans la route courante
        if charge + dem <= capacite:
            route_courante.append(client)
            charge += dem
        else:
            # TODO 3 : fermer la route courante -> [dep] + clients + [dep]
            if len(route_courante) > 0:
                route_complete = [depot_id]
                j = 0
                while j < len(route_courante):
                    route_complete.append(route_courante[j])
                    j += 1
                route_complete.append(depot_id)
                routes.append(route_complete)

            # TODO 4 : démarrer une nouvelle route avec ce client
            route_courante = [client]
            charge = dem

        i += 1

    # TODO 5 : après la boucle, si route_courante non vide -> la fermer et l’ajouter
    if len(route_courante) > 0:
        route_complete = [depot_id]
        j = 0
        while j < len(route_courante):
            route_complete.append(route_courante[j])
            j += 1
        route_complete.append(depot_id)
        routes.append(route_complete)

    return routes

def cout_total_routes(routes, coords):
    total = 0.0
    i = 0
    while i < len(routes):
        total += calculer_cout(routes[i], coords)
        i += 1
    return total

def tracer_routes(fichier, routes, titre="Solution gloutonne faisable"):
    coords = lire_coordonnees(fichier)
    depots = lire_depots(fichier)

    # --- points + étiquettes (même logique que ton tracer)
    clients = []
    for ident in coords.keys():
        est_depot = False
        k = 0
        while k < len(depots):
            if ident == depots[k]:
                est_depot = True
                break
            k += 1
        if not est_depot:
            clients.append(ident)

    x_clients, y_clients = [], []
    i = 0
    while i < len(clients):
        x, y = coords[clients[i]]
        x_clients.append(x); y_clients.append(y)
        i += 1

    x_depots, y_depots = [], []
    i = 0
    while i < len(depots):
        x, y = coords[depots[i]]
        x_depots.append(x); y_depots.append(y)
        i += 1

    plt.figure()
    plt.scatter(x_clients, y_clients, c='blue', label='Clients')
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    for ident, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(ident), fontsize=8, color="black")

    # --- tracer chaque route
    couleurs = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'cyan', 'magenta']

    r = 0
    while r < len(routes):
        route = routes[r]
        couleur = couleurs[r % len(couleurs)]  # avance la couleur à chaque route

        i = 0
        while i < len(route) - 1:
            x1, y1 = coords[route[i]]
            x2, y2 = coords[route[i+1]]
            plt.plot([x1, x2], [y1, y2], '-', linewidth=2, color=couleur)
            i += 1

        r += 1

    plt.title(titre)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def generer_voisins_relocate(routes, demandes, capacite):
    voisins = []

    i = 0
    while i < len(routes):
        route_i = routes[i]

        k = 1
        while k < len(route_i) - 1:
            client = route_i[k]

            j = 0
            while j < len(routes):
                if j != i:
                    route_j = routes[j]

                    pos = 1
                    while pos < len(route_j):
                        demande_client = demandes.get(client, 0)

                        # calcul de la charge de la route j
                        charge_j = 0
                        m = 1
                        while m < len(route_j) - 1:
                            charge_j += demandes.get(route_j[m], 0)
                            m += 1

                        # Vérification capacité
                        if charge_j + demande_client <= capacite:
                            # Construction du voisin
                            nv_routes = []

                            # Copier toutes les routes
                            r = 0
                            while r < len(routes):
                                nv_routes.append(routes[r][:])
                                r += 1

                            # Retirer le client de route_i
                            nv_routes[i].pop(k)

                            # Insérer dans route_j à la position pos
                            nv_routes[j].insert(pos, client)

                            voisins.append(nv_routes)

                        pos += 1

                j += 1
            k += 1
        i += 1

    return voisins

def meilleur_voisin_routes(voisins, coords):
    meilleur = None
    meilleur_cout = float("inf")

    i = 0
    while i < len(voisins):
        v = voisins[i]
        c = cout_total_routes(v, coords)
        if c < meilleur_cout:
            meilleur_cout = c
            meilleur = v
        i += 1
    
    return meilleur, meilleur_cout

def meilleur_voisin_routes(voisins, coords):
    meilleur = None
    meilleur_cout = float("inf")

    i = 0
    while i < len(voisins):
        v = voisins[i]
        c = cout_total_routes(v, coords)
        if c < meilleur_cout:
            meilleur_cout = c
            meilleur = v
        i += 1

    return meilleur, meilleur_cout


coords = lire_coordonnees(choix_fichier)
demandes = lire_demandes(choix_fichier)
depots = lire_depots(choix_fichier)
capacite = lire_capacite(choix_fichier)
depot_id = depots[0]

# 1) On part d'une solution VRP faisable
tour_test = creer_solution_initiale(choix_fichier)
routes_test = split_par_capacite(tour_test, demandes, capacite, depot_id)

print("Coût initial :", cout_total_routes(routes_test, coords))
tracer_routes(choix_fichier, routes_test, titre="Solution avant relocate")

# 2) Générer un voisin relocate
cout_init = cout_total_routes(routes_test, coords)
print("Coût initial :", cout_init)
tracer_routes(choix_fichier, routes_test, titre="Solution avant relocate")

voisins = generer_voisins_relocate(routes_test, demandes, capacite)
print("Nombre de voisins relocate générés :", len(voisins))

if len(voisins) > 0:
    best_vrp, best_cout = meilleur_voisin_routes(voisins, coords)
    if best_cout < cout_init:
        print("✅ Amélioration trouvée !")
        print("Nouveau coût :", best_cout)
        print("Gain :", cout_init - best_cout)
        tracer_routes(choix_fichier, best_vrp, titre="Après relocate (meilleur voisin)")
    else:
        print("⚠️ Aucun voisin n'améliore la solution (ce tour).")
else:
    print("⚠️ Aucun voisin généré.")

