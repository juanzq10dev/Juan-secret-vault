from dataclasses import dataclass

import pandas as pd

from src.domain.pokemon.pokemon import Pokemon, Stats
from src.domain.pokemon.types import PokemonType
from src.repositories.dataset_repository import PokemonColumn
from src.services.cluster_service import (
    ClusterRole,
    ClusterService,
    STAT_COLS,
    STAT_VALUES,
)


@dataclass(frozen=True)
class ClassifiedPokemon:
    pokemon: Pokemon
    cluster: int
    role: str


@dataclass(frozen=True)
class Team:
    members: list[ClassifiedPokemon]
    roles_covered: list[str]


class TeamBuilderService:

    def __init__(self, cluster_service: ClusterService) -> None:
        self._cluster_service = cluster_service

    def to_classified_pokemon(
        self, row: pd.Series, cluster_roles: dict[int, ClusterRole]
    ) -> ClassifiedPokemon:
        cluster_id = int(row["cluster"])
        pokemon = Pokemon(
            name=row[PokemonColumn.NAME.value],
            type1=PokemonType(row[PokemonColumn.TYPE1.value]),
            type2=(
                PokemonType(row[PokemonColumn.TYPE2.value])
                if pd.notna(row[PokemonColumn.TYPE2.value])
                else None
            ),
            stats=Stats(
                hp=int(row[PokemonColumn.HP.value]),
                attack=int(row[PokemonColumn.ATTACK.value]),
                defense=int(row[PokemonColumn.DEFENSE.value]),
                sp_attack=int(row[PokemonColumn.SP_ATTACK.value]),
                sp_defense=int(row[PokemonColumn.SP_DEFENSE.value]),
                speed=int(row[PokemonColumn.SPEED.value]),
                total=int(row[STAT_VALUES].sum()),
            ),
        )
        return ClassifiedPokemon(
            pokemon=pokemon,
            cluster=cluster_id,
            role=cluster_roles[cluster_id].role,
        )

    def suggest_team(
        self, df: pd.DataFrame, n_clusters: int, starter_name: str
    ) -> Team:
        cluster_roles = self._cluster_service.compute_roles(df, n_clusters)

        starter_matches = df[
            df[PokemonColumn.NAME.value].str.lower() == starter_name.lower()
        ]
        if starter_matches.empty:
            raise ValueError(f"Pokemon '{starter_name}' not found.")

        starter_row = starter_matches.iloc[0]
        starter_cluster = int(starter_row["cluster"])

        team_rows = [starter_row]
        for cluster_id in range(n_clusters):
            if len(team_rows) >= 6:
                break
            if cluster_id == starter_cluster:
                continue
            cluster_df = df[df["cluster"] == cluster_id].copy()
            cluster_df["_total"] = cluster_df[STAT_VALUES].sum(axis=1)
            best = cluster_df.sort_values("_total", ascending=False).iloc[0]
            team_rows.append(best)

        members = [self.to_classified_pokemon(row, cluster_roles) for row in team_rows]
        roles_covered = sorted({m.role for m in members})

        return Team(members=members, roles_covered=roles_covered)
