import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.domain.pokemon.types import PokemonType, TypeChart
from src.repositories.dataset_repository import PokemonColumn, PokemonCsvRepository
from src.repositories.type_chart_repository import JsonTypeChartRepository


STAT_COLS = [
    PokemonColumn.HP,
    PokemonColumn.ATTACK,
    PokemonColumn.DEFENSE,
    PokemonColumn.SP_ATTACK,
    PokemonColumn.SP_DEFENSE,
    PokemonColumn.SPEED,
]
RATIO_COLS = [f"{col.value}_ratio" for col in STAT_COLS]

# One bias dict per archetype — any stat combination is valid
STAT_BIASES: list[dict[PokemonColumn, float]] = [
    {PokemonColumn.HP: 3.0},  # Tank
    {PokemonColumn.ATTACK: 3.0},  # Physical Sweeper
    {PokemonColumn.DEFENSE: 3.0},  # Defensive Wall
    {PokemonColumn.SP_ATTACK: 3.0},  # Special Attacker
    {PokemonColumn.SP_DEFENSE: 3.0},  # Special Wall
    {PokemonColumn.SPEED: 3.0},  # Fast Attacker
]


def _build_stat_centroids() -> np.ndarray:
    return np.array(
        [[biases.get(col, 0.0) for col in STAT_COLS] for biases in STAT_BIASES],
        dtype=float,
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path = Path("data/pokemon_hgss.csv")
    output_path: Path = Path("model.pkl")
    type_chart_path: Path = Path("data/type_chart.json")
    type_weight: float = 0.5
    weakness_weight: float = 1.2
    random_state: int = 42

    @property
    def n_clusters(self) -> int:
        return len(STAT_BIASES)

    def __post_init__(self) -> None:
        if self.n_clusters != len(STAT_BIASES):
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must match len(STAT_BIASES) ({len(STAT_BIASES)})."
            )


@dataclass
class TrainingResult:
    model: KMeans
    scaler: StandardScaler
    df: pd.DataFrame
    type_cols: list[str]
    ratio_cols: list[str]


# ── KMeans training adapter ───────────────────────────────────────────────────


class KMeansTrainingAdapter:
    """Encapsulates all KMeans-specific feature engineering and model fitting."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def build_features(
        self,
        df: pd.DataFrame,
        type_chart: TypeChart,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, StandardScaler, list[str]]:
        df = self._compute_stat_ratios(df)
        type_matrix, type_cols = self._encode_types(df)
        weakness_matrix = self._build_weakness_matrix(df, type_chart)

        if scaler is None:
            scaler = StandardScaler()
            ratios_norm = scaler.fit_transform(df[RATIO_COLS])
        else:
            ratios_norm = scaler.transform(df[RATIO_COLS])

        features = np.hstack(
            [
                ratios_norm,
                type_matrix * self._config.type_weight,
                weakness_matrix * self._config.weakness_weight,
            ]
        )
        return features, scaler, type_cols

    def fit(self, features: np.ndarray) -> KMeans:
        stat_centroids = _build_stat_centroids()
        n_extra = features.shape[1] - len(STAT_COLS)
        centroids = np.hstack(
            [stat_centroids, np.zeros((self._config.n_clusters, n_extra))]
        )
        model = KMeans(
            n_clusters=self._config.n_clusters,
            init=centroids,
            n_init=1,
            random_state=self._config.random_state,
        )
        model.fit(features)
        return model

    def _compute_stat_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        stat_values = [col.value for col in STAT_COLS]
        df["_total"] = df[stat_values].sum(axis=1)
        for col in STAT_COLS:
            df[f"{col.value}_ratio"] = df[col.value] / df["_total"]
        return df

    def _encode_types(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        type1_dummies = pd.get_dummies(df[PokemonColumn.TYPE1.value], prefix="type")
        type2_dummies = pd.get_dummies(
            df[PokemonColumn.TYPE2.value].fillna("None"), prefix="type"
        ).drop(columns="type_None", errors="ignore")
        type_dummies = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)
        return type_dummies.reset_index(drop=True).values, type_dummies.columns.tolist()

    def _build_weakness_matrix(
        self, df: pd.DataFrame, type_chart: TypeChart
    ) -> np.ndarray:
        all_types = list(type_chart.keys())
        rows = []
        for _, row in df.iterrows():
            type1 = PokemonType(row[PokemonColumn.TYPE1.value])
            type2 = (
                PokemonType(row[PokemonColumn.TYPE2.value])
                if pd.notna(row[PokemonColumn.TYPE2.value])
                else None
            )
            multipliers = [
                type_chart[attacker].get_effectiveness_against(type1)
                * (
                    type_chart[attacker].get_effectiveness_against(type2)
                    if type2
                    else 1.0
                )
                for attacker in all_types
            ]
            rows.append(multipliers)
        return np.array(rows)


# ── Persistence ───────────────────────────────────────────────────────────────


def save_model(result: TrainingResult, output_path: Path) -> None:
    payload = {
        "model": result.model,
        "scaler": result.scaler,
        "type_cols": result.type_cols,
        "ratio_cols": result.ratio_cols,
        "df": result.df,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def train(config: TrainingConfig = TrainingConfig()) -> None:
    type_chart = JsonTypeChartRepository(config.type_chart_path).load()
    df = PokemonCsvRepository(config.data_path).load()

    adapter = KMeansTrainingAdapter(config)
    features, scaler, type_cols = adapter.build_features(df, type_chart)
    model = adapter.fit(features)

    df["cluster"] = model.labels_
    df = adapter._compute_stat_ratios(df)

    keep_cols = (
        [PokemonColumn.NAME.value, PokemonColumn.TYPE1.value, PokemonColumn.TYPE2.value]
        + [col.value for col in STAT_COLS]
        + RATIO_COLS
        + ["cluster"]
    )
    result = TrainingResult(
        model=model,
        scaler=scaler,
        df=df[keep_cols].reset_index(drop=True),
        type_cols=type_cols,
        ratio_cols=RATIO_COLS,
    )

    save_model(result, config.output_path)
    print(f"Model trained on {len(df)} rows → saved to {config.output_path}")
    _print_cluster_summary(result.df)


def _print_cluster_summary(df: pd.DataFrame) -> None:
    stat_values = [col.value for col in STAT_COLS]
    print("\nCluster summary:")
    for cluster_id in sorted(df["cluster"].unique()):
        group = df[df["cluster"] == cluster_id]
        dominant = str(group[RATIO_COLS].mean().idxmax()).replace("_ratio", "")
        avg_total = group[stat_values].sum(axis=1).mean()
        print(
            f"  Cluster {cluster_id}: {len(group):3d} rows | "
            f"avg Total={avg_total:.0f} | dominant stat={dominant}"
        )


if __name__ == "__main__":
    train()
