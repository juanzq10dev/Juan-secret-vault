# Gen IV type effectiveness chart (HeartGold/SoulSilver — no Fairy type)
# TYPE_CHART[attacker][defender] = damage multiplier
# Only non-1.0 values are stored; anything missing defaults to 1.0

TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal": {"Rock": 0.5, "Steel": 0.5, "Ghost": 0},
    "Fire": {
        "Grass": 2,
        "Ice": 2,
        "Bug": 2,
        "Steel": 2,
        "Fire": 0.5,
        "Water": 0.5,
        "Rock": 0.5,
        "Dragon": 0.5,
    },
    "Water": {
        "Fire": 2,
        "Ground": 2,
        "Rock": 2,
        "Water": 0.5,
        "Grass": 0.5,
        "Dragon": 0.5,
    },
    "Electric": {
        "Water": 2,
        "Flying": 2,
        "Electric": 0.5,
        "Grass": 0.5,
        "Dragon": 0.5,
        "Ground": 0,
    },
    "Grass": {
        "Water": 2,
        "Ground": 2,
        "Rock": 2,
        "Fire": 0.5,
        "Grass": 0.5,
        "Poison": 0.5,
        "Flying": 0.5,
        "Bug": 0.5,
        "Dragon": 0.5,
        "Steel": 0.5,
    },
    "Ice": {
        "Grass": 2,
        "Ground": 2,
        "Flying": 2,
        "Dragon": 2,
        "Water": 0.5,
        "Ice": 0.5,
        "Steel": 0.5,
    },
    "Fighting": {
        "Normal": 2,
        "Ice": 2,
        "Rock": 2,
        "Dark": 2,
        "Steel": 2,
        "Poison": 0.5,
        "Bug": 0.5,
        "Psychic": 0.5,
        "Flying": 0.5,
        "Ghost": 0,
    },
    "Poison": {
        "Grass": 2,
        "Poison": 0.5,
        "Ground": 0.5,
        "Rock": 0.5,
        "Ghost": 0.5,
        "Steel": 0,
    },
    "Ground": {
        "Fire": 2,
        "Electric": 2,
        "Poison": 2,
        "Rock": 2,
        "Steel": 2,
        "Grass": 0.5,
        "Bug": 0.5,
        "Flying": 0,
    },
    "Flying": {
        "Grass": 2,
        "Fighting": 2,
        "Bug": 2,
        "Electric": 0.5,
        "Rock": 0.5,
        "Steel": 0.5,
    },
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Steel": 0.5, "Dark": 0},
    "Bug": {
        "Grass": 2,
        "Psychic": 2,
        "Dark": 2,
        "Fire": 0.5,
        "Fighting": 0.5,
        "Flying": 0.5,
        "Ghost": 0.5,
        "Steel": 0.5,
    },
    "Rock": {
        "Fire": 2,
        "Ice": 2,
        "Flying": 2,
        "Bug": 2,
        "Fighting": 0.5,
        "Ground": 0.5,
        "Steel": 0.5,
    },
    "Ghost": {"Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5, "Normal": 0},
    "Dragon": {"Dragon": 2, "Steel": 0.5},
    "Dark": {"Psychic": 2, "Ghost": 2, "Fighting": 0.5, "Dark": 0.5, "Steel": 0.5},
    "Steel": {
        "Ice": 2,
        "Rock": 2,
        "Fire": 0.5,
        "Water": 0.5,
        "Electric": 0.5,
        "Steel": 0.5,
    },
}

ALL_TYPES = list(TYPE_CHART.keys())


def _multiplier(attacker: str, defender: str) -> float:
    return TYPE_CHART.get(attacker, {}).get(defender, 1.0)


def defending_multipliers(type1: str, type2: str = "") -> dict[str, float]:
    """
    Returns a dict of {attacking_type: multiplier} for a pokemon
    with the given type(s). Multiplier is the product of both types.
    """
    result = {}
    for attacker in ALL_TYPES:
        m = _multiplier(attacker, type1)
        if type2:
            m *= _multiplier(attacker, type2)
        result[attacker] = m
    return result


def get_weaknesses(type1: str, type2: str = "") -> list[str]:
    """Attacking types that deal more than 1x damage."""
    return [t for t, m in defending_multipliers(type1, type2).items() if m > 1]


def get_resistances(type1: str, type2: str = "") -> list[str]:
    """Attacking types that deal less than 1x damage (including immunities)."""
    return [t for t, m in defending_multipliers(type1, type2).items() if m < 1]


def team_weakness_score(team: list[dict]) -> dict[str, float]:
    """
    Given a list of pokemon dicts (with 'type1' and 'type2' keys),
    returns the total damage multiplier each attacking type deals
    summed across the whole team. Higher = team is more exposed to that type.
    """
    scores: dict[str, float] = {t: 0.0 for t in ALL_TYPES}
    for pokemon in team:
        multipliers = defending_multipliers(
            pokemon["type1"], pokemon.get("type2") or ""
        )
        for attacker, m in multipliers.items():
            scores[attacker] += m
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    # Quick sanity check
    print("Umbreon (Dark) weaknesses:  ", get_weaknesses("Dark"))
    print("Umbreon (Dark) resistances: ", get_resistances("Dark"))
    print()
    print("Gyarados (Water/Flying) weaknesses:  ", get_weaknesses("Water", "Flying"))
    print("Gyarados (Water/Flying) resistances: ", get_resistances("Water", "Flying"))
