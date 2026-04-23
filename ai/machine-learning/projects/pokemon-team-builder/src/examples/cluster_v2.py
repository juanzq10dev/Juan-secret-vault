import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.domain.pokemon.types import PokemonType
from src.repositories.type_chart_repository import JsonTypeChartRepository

_type_chart = JsonTypeChartRepository().load()
ALL_TYPES = list(_type_chart.keys())


def defending_multipliers(type1: str, type2: str = "") -> dict:
    t1 = PokemonType(type1)
    t2 = PokemonType(type2) if type2 else None
    return {
        attacker: _type_chart[attacker].get_effectiveness_against(t1)
        * (_type_chart[attacker].get_effectiveness_against(t2) if t2 else 1.0)
        for attacker in ALL_TYPES
    }


# 1. Cargar datos
df = pd.read_csv("data/pokemon_hgss.csv")

# 2. Construir features: multiplicadores defensivos por tipo atacante
# Cada pokemon → 17 valores (uno por tipo atacante)
# 0 = inmune, 0.25/0.5 = resistente, 1 = normal, 2/4 = débil
weakness_rows = []
for _, row in df.iterrows():
    t2 = row["Type 2"] if pd.notna(row["Type 2"]) else ""
    multipliers = defending_multipliers(row["Type 1"], t2)
    weakness_rows.append([multipliers[t] for t in ALL_TYPES])

features = np.array(weakness_rows)

# 3. K-Means — 6 clusters de perfiles defensivos similares
# Sin escalar: los valores (0, 0.5, 1, 2, 4) ya son comparables entre sí
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(features)
df["cluster"] = kmeans.labels_

# 4. Resumen — qué tipos atacantes son la mayor debilidad promedio de cada cluster
print("Cluster summary — avg weakness per attacking type:\n")
for cluster_id in sorted(df["cluster"].unique()):
    group_idx = df[df["cluster"] == cluster_id].index
    avg = features[group_idx].mean(axis=0)
    worst = sorted(zip(ALL_TYPES, avg), key=lambda x: x[1], reverse=True)[:3]
    best = sorted(zip(ALL_TYPES, avg), key=lambda x: x[1])[:3]
    types = df.loc[group_idx, "Type 1"].value_counts().index[0]
    print(
        f"  Cluster {cluster_id} ({len(group_idx):3d} pokemon | dominant type: {types})"
    )
    print(f"    Weak to:     {', '.join(f'{t}({v:.1f}x)' for t, v in worst)}")
    print(f"    Resists:     {', '.join(f'{t}({v:.1f}x)' for t, v in best)}")

# 5. Reducir a 2D con PCA para graficar
pca = PCA(n_components=2)
coords = pca.fit_transform(features)

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

plt.figure(figsize=(10, 7))
for i in range(6):
    mask = df["cluster"] == i
    plt.scatter(
        coords[mask, 0], coords[mask, 1], c=colors[i], label=f"Cluster {i}", alpha=0.7
    )
    for idx in df[mask].index:
        plt.annotate(
            df.loc[idx, "Name"], (coords[idx, 0], coords[idx, 1]), fontsize=5, alpha=0.6
        )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.title("Clusters de Pokemon por perfil defensivo (PCA 2D)")
plt.legend()
plt.tight_layout()
plt.savefig("data/clusters_v2.png", dpi=150)
plt.show()
