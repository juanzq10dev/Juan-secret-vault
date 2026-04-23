from dataclasses import dataclass
from enum import Enum


DEFAULT_EFFECTIVENESS = 1.0


class PokemonType(str, Enum):
    NORMAL = "Normal"
    FIRE = "Fire"
    WATER = "Water"
    ELECTRIC = "Electric"
    GRASS = "Grass"
    ICE = "Ice"
    FIGHTING = "Fighting"
    POISON = "Poison"
    GROUND = "Ground"
    FLYING = "Flying"
    PSYCHIC = "Psychic"
    BUG = "Bug"
    ROCK = "Rock"
    GHOST = "Ghost"
    DRAGON = "Dragon"
    DARK = "Dark"
    STEEL = "Steel"


TypeChart = dict[PokemonType, "TypeEffectiveness"]


@dataclass(frozen=True)
class TypeEffectiveness:
    type: PokemonType
    effectiveness: dict[PokemonType, float]

    def get_effectiveness_against(self, defender: PokemonType) -> float:
        """Returns the damage multiplier this type deals against the defender."""
        return self.effectiveness.get(defender, DEFAULT_EFFECTIVENESS)

    def get_weaknesses(self) -> list[PokemonType]:
        """Types that this type deals more than 1x damage to."""
        return [t for t, m in self.effectiveness.items() if m > DEFAULT_EFFECTIVENESS]

    def get_resistances(self) -> list[PokemonType]:
        """Types that this type deals less than 1x damage to (including immunities)."""
        return [t for t, m in self.effectiveness.items() if m < DEFAULT_EFFECTIVENESS]
