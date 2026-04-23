import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# 2. Calcular proporciones — cada stat dividida entre el total del pokemon
stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
df["_total"] = df[stat_cols].sum(axis=1)
ratio_cols = [f"{col}_ratio" for col in stat_cols]
for col in stat_cols:
    df[f"{col}_ratio"] = df[col] / df["_total"]

# 3. Encodear tipos — combinar Type 1 y Type 2 en las mismas columnas
type1_dummies = pd.get_dummies(df["Type 1"], prefix="type")
type2_dummies = pd.get_dummies(df["Type 2"].fillna("None"), prefix="type").drop(
    columns="type_None", errors="ignore"
)
type_dummies = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)

# 4. Weakness rows — multiplicadores defensivos por tipo atacante (17 valores)
weakness_rows = []
for _, row in df.iterrows():
    t2 = row["Type 2"] if pd.notna(row["Type 2"]) else ""
    multipliers = defending_multipliers(row["Type 1"], t2)
    weakness_rows.append([multipliers[t] for t in ALL_TYPES])
weakness_matrix = np.array(weakness_rows)

# 5. Normalizar ratios y concatenar con tipos + weakness rows
scaler = StandardScaler()
stats_norm = scaler.fit_transform(df[ratio_cols])
TYPE_WEIGHT = 0.5
WEAKNESS_WEIGHT = 1.2
features = np.hstack(
    [
        stats_norm,
        type_dummies.reset_index(drop=True).values * TYPE_WEIGHT,
        weakness_matrix * WEAKNESS_WEIGHT,
    ]
)

# 6. Aplicar K-Means con centroides sesgados — un arquetipo por stat
# Orden: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed + type cols + weakness cols (todas en 0)
n_extra_cols = features.shape[1] - 6
stat_centroids = np.array(
    [
        [3, 0, 0, 0, 0, 0],  # Tank        → HP alto
        [0, 3, 0, 0, 0, 0],  # Physical    → Attack alto
        [0, 0, 3, 0, 0, 0],  # Defensive   → Defense alta
        [0, 0, 0, 3, 0, 0],  # Special Atk → Sp. Atk alto
        [0, 0, 0, 0, 3, 0],  # Special Def → Sp. Def alta
        [0, 0, 0, 0, 0, 3],  # Fast        → Speed alta
    ],
    dtype=float,
)
centroids = np.hstack([stat_centroids, np.zeros((6, n_extra_cols))])

kmeans = KMeans(n_clusters=6, init=centroids, n_init=1, random_state=42)
kmeans.fit(features)

# 6. Asignar el cluster a cada Pokémon
df["cluster"] = kmeans.labels_

# Ver qué Pokémon quedaron en cada grupo (ordenados por Total descendente)
df_sorted = df.sort_values(["cluster", "_total"], ascending=[True, False])
print(
    df_sorted.groupby("cluster")[
        ["Name", "Type 1", "Type 2", "HP", "Attack", "Defense", "Speed"]
    ].head(3)
)

# 7. Reducir a 2D con PCA para graficar
pca = PCA(n_components=2)
coords = pca.fit_transform(features)

plt.figure(figsize=(10, 7))
colors = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#e91e63",
    "#00bcd4",
    "#ff5722",
    "#607d8b",
    "#8bc34a",
    "#ff9800",
    "#673ab7",
    "#009688",
    "#f06292",
    "#795548",
]

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
plt.title("Clusters de Pokémon (PCA 2D)")
plt.legend()
plt.tight_layout()
plt.savefig("data/clusters.png", dpi=150)
plt.show()
