from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd


class PokemonColumn(str, Enum):
    NAME = "name"
    TYPE1 = "type1"
    TYPE2 = "type2"
    HP = "hp"
    ATTACK = "attack"
    DEFENSE = "defense"
    SP_ATTACK = "sp_attack"
    SP_DEFENSE = "sp_defense"
    SPEED = "speed"


STANDARD_COLUMNS = list(PokemonColumn)

# Default mapping for the current CSV (pokemon_hgss.csv)
DEFAULT_COLUMN_MAPPING: dict[str, PokemonColumn] = {
    "Name": PokemonColumn.NAME,
    "Type 1": PokemonColumn.TYPE1,
    "Type 2": PokemonColumn.TYPE2,
    "HP": PokemonColumn.HP,
    "Attack": PokemonColumn.ATTACK,
    "Defense": PokemonColumn.DEFENSE,
    "Sp. Atk": PokemonColumn.SP_ATTACK,
    "Sp. Def": PokemonColumn.SP_DEFENSE,
    "Speed": PokemonColumn.SPEED,
}


class DatasetRepository(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Returns a DataFrame with standardized column names."""
        pass


@dataclass
class PokemonCsvRepository(DatasetRepository):
    path: Path
    column_mapping: dict[str, PokemonColumn] = field(
        default_factory=lambda: DEFAULT_COLUMN_MAPPING
    )

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        self._validate_columns(df)
        df = df.rename(columns={k: v.value for k, v in self.column_mapping.items()})
        return df[[col.value for col in STANDARD_COLUMNS]]

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = set(self.column_mapping.keys()) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV at '{self.path}' is missing expected columns: {missing}. "
                f"Update the column_mapping to match your CSV."
            )
