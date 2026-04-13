import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from api.type_chart import ALL_TYPES, defending_multipliers

STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
TYPE_WEIGHT = 0.5
WEAKNESS_WEIGHT = 1.2

# One biased centroid per role — each row seeds one archetype
_STAT_CENTROIDS = np.array(
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


def _build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str], list[str]]:
    # Stat ratios — removes power level, keeps role shape
    df = df.copy()
    df["_total"] = df[STAT_COLS].sum(axis=1)
    ratio_cols = [f"{col}_ratio" for col in STAT_COLS]
    for col in STAT_COLS:
        df[f"{col}_ratio"] = df[col] / df["_total"]

    # Type encoding
    type1_dummies = pd.get_dummies(df["Type 1"], prefix="type")
    type2_dummies = pd.get_dummies(df["Type 2"].fillna("None"), prefix="type").drop(
        columns="type_None", errors="ignore"
    )
    type_dummies = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)
    type_cols = type_dummies.columns.tolist()

    # Defensive weakness matrix (17 values per pokemon)
    weakness_rows = []
    for _, row in df.iterrows():
        t2 = row["Type 2"] if pd.notna(row["Type 2"]) else ""
        multipliers = defending_multipliers(row["Type 1"], t2)
        weakness_rows.append([multipliers[t] for t in ALL_TYPES])
    weakness_matrix = np.array(weakness_rows)

    # Normalize ratios
    if scaler is None:
        scaler = StandardScaler()
        ratios_norm = scaler.fit_transform(df[ratio_cols])
    else:
        ratios_norm = scaler.transform(df[ratio_cols])

    features = np.hstack(
        [
            ratios_norm,
            type_dummies.reset_index(drop=True).values * TYPE_WEIGHT,
            weakness_matrix * WEAKNESS_WEIGHT,
        ]
    )
    return features, scaler, type_cols, ratio_cols


def train(
    data_path: str = "data/pokemon_hgss.csv", output_path: str = "model.pkl"
) -> None:
    df = pd.read_csv(data_path)

    features, scaler, type_cols, ratio_cols = _build_features(df)

    # Pad biased centroids with zeros for type + weakness columns
    n_extra = features.shape[1] - 6
    centroids = np.hstack([_STAT_CENTROIDS, np.zeros((6, n_extra))])

    kmeans = KMeans(n_clusters=6, init=centroids, n_init=1, random_state=42)
    kmeans.fit(features)
    df["cluster"] = kmeans.labels_

    # Add ratio cols to df so main.py can use them for role detection
    df["_total"] = df[STAT_COLS].sum(axis=1)
    for col in STAT_COLS:
        df[f"{col}_ratio"] = df[col] / df["_total"]

    keep_cols = (
        ["Name", "Type 1", "Type 2"]
        + STAT_COLS
        + [f"{col}_ratio" for col in STAT_COLS]
        + ["cluster"]
    )
    payload = {
        "model": kmeans,
        "scaler": scaler,
        "type_cols": type_cols,
        "ratio_cols": ratio_cols,
        "df": df[keep_cols].reset_index(drop=True),
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Model trained on {len(df)} pokemon → saved to {output_path}")
    _print_cluster_summary(df, ratio_cols)


def _print_cluster_summary(df: pd.DataFrame, ratio_cols: list[str]) -> None:
    print("\nCluster summary:")
    for cluster_id in sorted(df["cluster"].unique()):
        group = df[df["cluster"] == cluster_id]
        dominant = str(group[ratio_cols].mean().idxmax()).replace("_ratio", "")
        avg_total = group[STAT_COLS].sum(axis=1).mean()
        print(
            f"  Cluster {cluster_id}: {len(group):3d} pokemon | "
            f"avg Total={avg_total:.0f} | dominant stat={dominant}"
        )


if __name__ == "__main__":
    train()
