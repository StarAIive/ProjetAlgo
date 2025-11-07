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

    # Identifier tous les clients (non-dépôts)
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

    # Fonction pour calculer la distance euclidienne entre deux points
    def distance(point1, point2):
        if point1 in coords and point2 in coords:
            x1, y1 = coords[point1]
            x2, y2 = coords[point2]
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return float('inf')

    # Algorithme glouton pour créer les routes
    routes = []
    clients_non_visites = clients[:]  # copie de la liste des clients
    
    while len(clients_non_visites) > 0:
        # Commencer une nouvelle route depuis le dépôt
        route_actuelle = [depot]
        capacite_actuelle = 0
        position_actuelle = depot
        
        while True:
            # Trouver le client non visité le plus proche qui respecte la contrainte de capacité
            meilleur_client = None
            meilleure_distance = float('inf')
            
            for client in clients_non_visites:
                # Vérifier la contrainte de capacité
                demande_client = demandes.get(client, 0)
                if capacites is None or (capacite_actuelle + demande_client <= capacites):
                    dist = distance(position_actuelle, client)
                    if dist < meilleure_distance:
                        meilleure_distance = dist
                        meilleur_client = client
            
            # Si aucun client ne peut être ajouté (contrainte de capacité ou plus de clients)
            if meilleur_client is None:
                break
            
            # Ajouter le meilleur client à la route
            route_actuelle.append(meilleur_client)
            capacite_actuelle += demandes.get(meilleur_client, 0)
            position_actuelle = meilleur_client
            clients_non_visites.remove(meilleur_client)
        
        # Retourner au dépôt pour fermer la route
        route_actuelle.append(depot)
        routes.append(route_actuelle)

    return routes

def calculer_cout(routes, coords):
    cout_total = 0.0

    # parcours des routes
    r = 0
    while r < len(routes):
        route = routes[r]
        i = 0
        # parcours des sommets de la route
        while i < len(route) - 1:
            u = route[i]
            v = route[i + 1]
            if (u in coords) and (v in coords):
                x1, y1 = coords[u]
                x2, y2 = coords[v]
                cout_total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            i += 1
        r += 1

    return cout_total
def generer_voisins_swap(routes):
    voisins = []
    # Limite dure pour éviter l'explosion (à ajuster si besoin)
    MAX_VOISINS = 10000

    # ---------- Swaps INTRA-route (adjacents uniquement) ----------
    r = 0
    while r < len(routes):
        route = routes[r]
        i = 1
        while i < len(route) - 2:
            j = i + 1  # adjacent
            # copie profonde manuelle
            v = []
            rr = 0
            while rr < len(routes):
                v.append(routes[rr][:])
                rr += 1
            # swap dans la route r
            tmp = v[r][i]
            v[r][i] = v[r][j]
            v[r][j] = tmp
            voisins.append(v)

            if len(voisins) >= MAX_VOISINS:
                return voisins

            i += 1
        r += 1

    # ---------- Swaps INTER-routes (adjacents uniquement) ----------
    r1 = 0
    while r1 < len(routes):
        r2 = r1 + 1
        while r2 < len(routes):
            route1 = routes[r1]
            route2 = routes[r2]
            i = 1
            while i < len(route1) - 1:
                j = i  # pairé par position relative (grossière mais rapide)
                if j < 1:
                    j = 1
                if j > len(route2) - 2:
                    j = len(route2) - 2
                if j >= 1 and j < len(route2) - 1:
                    # copie profonde manuelle
                    v = []
                    rr = 0
                    while rr < len(routes):
                        v.append(routes[rr][:])
                        rr += 1
                    # swap entre r1[i] et r2[j]
                    tmp = v[r1][i]
                    v[r1][i] = v[r2][j]
                    v[r2][j] = tmp
                    voisins.append(v)

                    if len(voisins) >= MAX_VOISINS:
                        return voisins
                i += 1
            r2 += 1
        r1 += 1

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

def amelioration_locale(routes, coords):
    amelioration = True
    cout_actuel = calculer_cout(routes, coords)
    while amelioration:
        voisins = generer_voisins_swap(routes)
        meilleur, meilleur_cout = meilleur_voisin(voisins, coords)
        if meilleur_cout < cout_actuel:
            routes = meilleur
            cout_actuel = meilleur_cout
        else:
            amelioration = False
    return routes, cout_actuel

def tabu_search(tour_initiale, coords, iterations=1000, tenure=7):
    # copie profonde initiale
    S = []
    r = 0
    while r < len(tour_initiale):
        S.append(tour_initiale[r][:])
        r += 1

    S_cost = calculer_cout(S, coords)
    best = []
    r = 0
    while r < len(S):
        best.append(S[r][:])
        r += 1
    best_cost = S_cost

    tabu = {}  # move_key -> expiration

    # dépôt (pour ignorer positions 0 et -1)
    depot = None
    if len(S) > 0 and len(S[0]) > 0:
        depot = S[0][0]

    # arrêt plateau si pas d'amélioration globale pendant trop longtemps
    PLATEAU_MAX = 500
    plateau = 0

    k = 1
    while k <= iterations:
        voisins = generer_voisins_swap(S)

        # FIRST-IMPROVEMENT: accepter le premier voisin admissible qui améliore S_cost
        accepted = False
        v_idx = 0
        best_cand = None
        best_cand_cost = float("inf")
        best_cand_move_key = None

        while v_idx < len(voisins):
            cand = voisins[v_idx]
            c = calculer_cout(cand, coords)

            # déduire move_key (couple de clients échangés, dépôts exclus)
            changed_values = set()
            rr = 0
            while rr < len(S):
                route_old = S[rr]
                route_new = cand[rr]
                ii = 0
                while ii < len(route_old):
                    if route_old[ii] != route_new[ii]:
                        a = route_old[ii]
                        b = route_new[ii]
                        if depot is not None:
                            if a != depot and b != depot and a != b:
                                changed_values.add(a)
                                changed_values.add(b)
                        else:
                            if a != b:
                                changed_values.add(a)
                                changed_values.add(b)
                    ii += 1
                rr += 1

            move_key = None
            if len(changed_values) >= 2:
                tmp = list(changed_values)
                tmp.sort()
                move_key = (tmp[0], tmp[1])

            est_tabu = False
            if move_key is not None and (move_key in tabu) and (tabu[move_key] >= k):
                est_tabu = True

            aspiration = (c < best_cost)

            # FIRST-IMPROVEMENT: on prend dès que c < S_cost si non-tabu (ou aspiration)
            if ((not est_tabu) and (c < S_cost)) or aspiration:
                S = []
                r = 0
                while r < len(cand):
                    S.append(cand[r][:])
                    r += 1
                S_cost = c

                if move_key is not None:
                    tabu[move_key] = k + tenure

                accepted = True
                break  # passer à l'itération suivante

            # sinon on garde quand même le meilleur candidat vu (pour ne pas stagner)
            if (not est_tabu) or aspiration:
                if c < best_cand_cost:
                    best_cand = cand
                    best_cand_cost = c
                    best_cand_move_key = move_key

            v_idx += 1

        if not accepted:
            # Aucun voisin strictement meilleur trouvé ; on bouge quand même vers le meilleur admissible vu
            if best_cand is None:
                break  # pas d'option admissible, on s'arrête
            S = []
            r = 0
            while r < len(best_cand):
                S.append(best_cand[r][:])
                r += 1
            S_cost = best_cand_cost
            if best_cand_move_key is not None:
                tabu[best_cand_move_key] = k + tenure

        # maj meilleur global + plateau
        if S_cost < best_cost:
            best = []
            r = 0
            while r < len(S):
                best.append(S[r][:])
                r += 1
            best_cost = S_cost
            plateau = 0
        else:
            plateau += 1
            if plateau >= PLATEAU_MAX:
                break

        k += 1

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
        x_clients.append(x)
        y_clients.append(y)

    x_depots, y_depots = [], []
    for did in depots:
        x, y = coords[did]
        x_depots.append(x)
        y_depots.append(y)

    # Plot points
    plt.figure()
    plt.scatter(x_clients, y_clients, c='blue', label='Clients')
    plt.scatter(x_depots, y_depots, c='red', marker='s', s=120, label='Dépôts')

    # Labels
    for i, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, color="black")

    # Tracer routes (couleur différente par route)
    if routes is not None and len(routes) > 0:

        r = 0
        while r < len(routes):
            route = routes[r]
            # Couleur: utiliser le cycle matplotlib 'C0', 'C1', ...
            couleur = f"C{r % 10}"
            # Tracer tous les arcs consécutifs de la route
            i = 0
            while i < len(route) - 1:
                u = route[i]
                v = route[i + 1]
                if (u in coords) and (v in coords):
                    x1, y1 = coords[u]
                    x2, y2 = coords[v]
                    # Ajouter le label une seule fois (sur le premier segment)
                    if i == 0:
                        plt.plot([x1, x2], [y1, y2], '-', linewidth=2.0, c=couleur, label=f"Camion {r+1}")
                    else:
                        plt.plot([x1, x2], [y1, y2], '-', linewidth=2.0, c=couleur)
                i += 1
            r += 1

    plt.title(titre)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


routes = creer_solution_initiale(choix_fichier)
print("Tournée initiale :", routes)
coords = lire_coordonnees(choix_fichier)

print("Capacité :", lire_capacite(choix_fichier))
print("Coût de la tournée initiale :", calculer_cout(routes, coords))

# Affichage initial
tracer_vrp(choix_fichier, routes, titre="Tournée initiale")

# Tabu Search
best_tabu, best_tabu_cost = tabu_search(routes, coords, iterations=1000, tenure=7)
print("Coût de la meilleure tournée (Tabu) :", best_tabu_cost)

# Affichage résultat
tracer_vrp(choix_fichier, best_tabu, titre="Tournée améliorée (Tabu)")
