import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Cargar datos
df = pd.read_csv("data/pokemon_hgss.csv")

# 2. Solo stats numéricas — sin tipos, añadir Total como feature
stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
df["Total"] = df[stat_cols].sum(axis=1)
features_cols = stat_cols + ["Total"]

# 3. Normalizar
scaler = StandardScaler()
features = scaler.fit_transform(df[features_cols])

# 4. K-Means con centroides sesgados — un arquetipo por stat
# Orden de columnas: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
import numpy as np

# 7 columnas: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Total
centroids = np.array(
    [
        [3, 0, 0, 0, 0, 0, 1],  # Tank        → HP alto
        [0, 3, 0, 0, 0, 0, 1],  # Physical    → Attack alto
        [0, 0, 3, 0, 0, 0, 1],  # Defensive   → Defense alta
        [0, 0, 0, 3, 0, 0, 1],  # Special Atk → Sp. Atk alto
        [0, 0, 0, 0, 3, 0, 1],  # Special Def → Sp. Def alta
        [0, 0, 0, 0, 0, 3, 1],  # Fast        → Speed alta
        [0, 0, 0, 0, 0, 0, 1],  # Strong      → Total alto (legendarios)
    ],
    dtype=float,
)

kmeans = KMeans(n_clusters=7, init=centroids, n_init=1, random_state=42)
kmeans.fit(features)
df["cluster"] = kmeans.labels_

# 5. Resumen por cluster
print("Cluster summary:")
for cluster_id in sorted(df["cluster"].unique()):
    group = df[df["cluster"] == cluster_id]
    dominant = group[features_cols].mean().idxmax()
    print(f"  Cluster {cluster_id}: {len(group):3d} pokemon | dominant stat={dominant}")

# 6. Reducir a 2D con PCA para graficar
pca = PCA(n_components=2)
coords = pca.fit_transform(features)

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#000"]

plt.figure(figsize=(10, 7))
for i in range(7):
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
plt.title("Clusters de Pokémon sin tipos (PCA 2D)")
plt.legend()
plt.tight_layout()
plt.savefig("data/clusters_2.png", dpi=150)
plt.show()
