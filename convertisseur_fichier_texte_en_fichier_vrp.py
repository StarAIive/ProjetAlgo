import os
import vrplib

# dossier où SONT tes .txt Solomon
SOURCE_DIR = "data2"

# dossier où TU veux stocker les .vrp
TARGET_DIR = r"C:\Users\DAVID\Desktop\CESI\Algorithme et optimisation combinatoire\projet\ProjetAlgo\ProjetAlgo2\ProjetAlgo\data2"

def read_solomon_txt(path):
    with open(path, "r") as f:
        lines = [line.rstrip() for line in f]

    name = lines[0].strip()

    # --- lire le bloc VEHICLE pour avoir nb véhicules et capacité ---
    vehicles = None
    capacity = None
    for i, line in enumerate(lines):
        if "NUMBER" in line and "CAPACITY" in line:
            parts = lines[i + 1].split()
            vehicles = int(parts[0])
            capacity = int(float(parts[1]))
            break
    if capacity is None:
        capacity = 999999
    if vehicles is None:
        vehicles = 1

    # --- trouver le début du tableau clients ---
    start_idx = None
    for i, line in enumerate(lines):
        if "CUST NO." in line.upper():
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Impossible de trouver la section clients dans {path}")

    node_coord = []
    demand = []
    time_window = []
    service_time = []

    for line in lines[start_idx:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue

        # id x y demand ready due service
        # on ignore l'id parce qu'en VRPLIB on va tout réindexer de 1 à n
        x = float(parts[1])
        y = float(parts[2])
        dem = int(float(parts[3]))
        ready = float(parts[4])
        due = float(parts[5])
        serv = float(parts[6])

        node_coord.append([x, y])
        demand.append(dem)
        time_window.append([ready, due])
        service_time.append(serv)

    # VRPLIB compte à partir de 1 et le dépôt est "1"
    dimension = len(node_coord)

    # on construit un VRAI dict VRPLIB (MAJUSCULES)
    instance_vrplib = {
        "NAME": name,
        "COMMENT": "converted from Solomon txt",
        "TYPE": "VRPTW",
        "VEHICLES": vehicles,
        "DIMENSION": dimension,
        "CAPACITY": capacity,
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        # sections
        "NODE_COORD_SECTION": node_coord,
        "DEMAND_SECTION": demand,
        "DEPOT_SECTION": [1],               # dépôt = premier noeud
        "TIME_WINDOW_SECTION": time_window,
        "SERVICE_TIME_SECTION": service_time,
    }
    return instance_vrplib


def convert_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    if not os.path.isdir(src_folder):
        print(f"Dossier source introuvable: {src_folder}")
        return

    txt_files = [f for f in os.listdir(src_folder) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"Aucun .txt dans {src_folder}")
        return

    for name in txt_files:
        src = os.path.join(src_folder, name)
        base, _ = os.path.splitext(name)
        dst = os.path.join(dst_folder, base + ".vrp")

        if os.path.exists(dst):
            print(f"{name} -> déjà converti ({dst})")
            continue

        try:
            inst = read_solomon_txt(src)
            # 👉 ordre correct : chemin puis instance
            vrplib.write_instance(dst, inst)
            print(f"{name} -> {dst} ✅")
        except Exception as e:
            print(f"{name} -> ERREUR : {e}")


if __name__ == "__main__":
    convert_folder(SOURCE_DIR, TARGET_DIR)
