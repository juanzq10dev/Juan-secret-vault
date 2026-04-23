from dataclasses import dataclass

import pandas as pd

from src.repositories.dataset_repository import PokemonColumn


STAT_COLS = [
    PokemonColumn.HP,
    PokemonColumn.ATTACK,
    PokemonColumn.DEFENSE,
    PokemonColumn.SP_ATTACK,
    PokemonColumn.SP_DEFENSE,
    PokemonColumn.SPEED,
]
STAT_VALUES = [col.value for col in STAT_COLS]
RATIO_COLS = [f"{col.value}_ratio" for col in STAT_COLS]

STAT_TO_ROLE: dict[str, str] = {
    PokemonColumn.HP.value: "Tank",
    PokemonColumn.ATTACK.value: "Physical Sweeper",
    PokemonColumn.DEFENSE.value: "Defensive Wall",
    PokemonColumn.SP_ATTACK.value: "Special Attacker",
    PokemonColumn.SP_DEFENSE.value: "Special Wall",
    PokemonColumn.SPEED.value: "Fast Attacker",
}


@dataclass(frozen=True)
class ClusterRole:
    cluster_id: int
    role: str
    dominant_stat: str
    avg_total: float
    size: int


class ClusterService:

    def compute_roles(
        self, df: pd.DataFrame, n_clusters: int
    ) -> dict[int, ClusterRole]:
        use_ratios = all(col in df.columns for col in RATIO_COLS)
        roles = {}
        for cluster_id in range(n_clusters):
            group = df[df["cluster"] == cluster_id]
            if use_ratios:
                dominant = str(group[RATIO_COLS].mean().idxmax()).replace("_ratio", "")
            else:
                dominant = str(group[STAT_VALUES].mean().idxmax())
            roles[cluster_id] = ClusterRole(
                cluster_id=cluster_id,
                role=STAT_TO_ROLE[dominant],
                dominant_stat=dominant,
                avg_total=round(group[STAT_VALUES].sum(axis=1).mean(), 1),
                size=len(group),
            )
        return roles
