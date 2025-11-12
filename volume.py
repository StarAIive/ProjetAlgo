import requests
import pandas as pd

# Jeu de données officiel OpenData Paris
DATASET = "comptage-multimodal-comptages"
BASE = "https://opendata.paris.fr/api/records/1.0/search/"

# On récupère beaucoup d’enregistrements pour agréger plusieurs capteurs
params = {
    "dataset": DATASET,
    "rows": 10000,          # limite par requête (on peut boucler pour plus)
    "sort": "date_comptage" # tri par date la plus récente
}

print("⏳ Récupération des données depuis OpenData Paris...")
resp = requests.get(BASE, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

rows = []
for r in data.get("records", []):
    f = r["fields"]
    # Champs corrects confirmés pour ce dataset
    t = f.get("date_comptage")
    compt = f.get("nb_vehicules")
    if t and compt is not None:
        rows.append({
            "t_1h": t,
            "v": compt,
            "id_site": f.get("identifiant")
        })

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError("Aucune donnée valide trouvée. Vérifie le dataset ou les champs.")

# Conversion des dates et agrégation par heure
df["t_1h"] = pd.to_datetime(df["t_1h"], errors="coerce")
df = df.dropna(subset=["t_1h"])
agg = df.groupby(df["t_1h"].dt.strftime("%Y-%m-%d %H:00:00")).v.sum().reset_index()
agg.columns = ["heure", "volume_total_veh"]

print("\n🚗 Volume horaire total (V) observé à Paris :\n")
print(agg.sort_values("heure", ascending=False).head(24))
