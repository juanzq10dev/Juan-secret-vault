from abc import ABC, abstractmethod
import json
from pathlib import Path

from src.domain.pokemon.types import PokemonType, TypeChart, TypeEffectiveness


class TypeChartRepository(ABC):

    @abstractmethod
    def load(self) -> TypeChart:
        pass


class JsonTypeChartRepository(TypeChartRepository):

    def __init__(self, path: Path = Path("data/type_chart.json")) -> None:
        self._path = path

    def load(self) -> TypeChart:
        raw: dict = json.loads(self._path.read_text())
        return {
            PokemonType(attacker): TypeEffectiveness(
                type=PokemonType(attacker),
                effectiveness={
                    PokemonType(defender): float(multiplier)
                    for defender, multiplier in matchups.items()
                },
            )
            for attacker, matchups in raw.items()
        }
