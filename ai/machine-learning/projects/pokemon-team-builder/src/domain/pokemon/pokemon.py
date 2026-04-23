from dataclasses import dataclass
from src.domain.pokemon.types import PokemonTypes


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
    type1: PokemonTypes
    type2: PokemonTypes | None
    stats: Stats
