import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Cargar datos
df = pd.read_csv("data/pokemon_hgss.csv")

# 2. Calcular proporciones — cada stat dividida entre el total del pokemon
# Esto elimina el nivel de poder y deja solo el rol
stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
df["_total"] = df[stat_cols].sum(axis=1)
ratio_cols = [f"{col}_ratio" for col in stat_cols]
for col in stat_cols:
    df[f"{col}_ratio"] = df[col] / df["_total"]

features = StandardScaler().fit_transform(df[ratio_cols])

# 3. K-Means con centroides sesgados — un arquetipo por stat
# Orden de columnas: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
centroids = np.array(
    [
        [3, 0, 0, 0, 0, 0],  # Tank        → HP alto
        [0, 3, 0, 0, 0, 0],  # Physical    → Attack alto
        [2, -1, 3, -1, 1, 0],  # Defensive   → Defense alta
        [0, 0, 0, 3, 0, 0],  # Special Atk → Sp. Atk alto
        [2, -1, 1, -1, 3, 0],  # Special Def → Sp. Def alta
        [0, 0, 0, 0, 0, 3],  # Fast        → Speed alta
    ],
    dtype=float,
)

kmeans = KMeans(n_clusters=6, init=centroids, n_init=1, random_state=42)
kmeans.fit(features)
df["cluster"] = kmeans.labels_

# 4. Resumen por cluster
print("Cluster summary:")
for cluster_id in sorted(df["cluster"].unique()):
    group = df[df["cluster"] == cluster_id]
    dominant = group[ratio_cols].mean().idxmax().replace("_ratio", "")
    print(f"  Cluster {cluster_id}: {len(group):3d} pokemon | dominant stat={dominant}")

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
plt.title("Clusters de Pokemon — stats only, biased centroids (PCA 2D)")
plt.legend()
plt.tight_layout()
plt.savefig("data/clusters_3.png", dpi=150)
plt.show()
