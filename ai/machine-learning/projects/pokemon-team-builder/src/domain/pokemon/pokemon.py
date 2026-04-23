from dataclasses import dataclass
from src.domain.pokemon.types import PokemonType


@dataclass
class Stats:
    hp: int
    attack: int
    defense: int
    sp_attack: int
    sp_defense: int
    speed: int
    total: int


@dataclass
class Pokemon:
    name: str
    type1: PokemonType
    type2: PokemonType | None
    stats: Stats
