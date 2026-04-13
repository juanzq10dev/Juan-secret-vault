import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = Path("model.pkl")

_model_data: dict = {}
_cluster_roles: dict[int, dict] = {}

_STAT_TO_ROLE = {
    "HP": "Tank",
    "Attack": "Physical Sweeper",
    "Defense": "Defensive Wall",
    "Sp. Atk": "Special Attacker",
    "Sp. Def": "Special Wall",
    "Speed": "Fast Attacker",
}


def _compute_cluster_roles(df: pd.DataFrame, n_clusters: int) -> dict[int, dict]:
    stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    ratio_cols = [f"{col}_ratio" for col in stat_cols]
    use_ratios = all(col in df.columns for col in ratio_cols)
    roles = {}
    for cluster_id in range(n_clusters):
        group = df[df["cluster"] == cluster_id]
        if use_ratios:
            dominant = str(group[ratio_cols].mean().idxmax()).replace("_ratio", "")
        else:
            dominant = str(group[stat_cols].mean().idxmax())
        roles[cluster_id] = {
            "role": _STAT_TO_ROLE[dominant],
            "dominant_stat": dominant,
            "avg_total": round(group[stat_cols].sum(axis=1).mean(), 1),
            "size": len(group),
        }
    return roles


def _load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError("model.pkl not found — run `python src/train.py` first.")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    _model_data.update(data)
    _cluster_roles.update(_compute_cluster_roles(data["df"], data["model"].n_clusters))


def _pokemon_dict(row: pd.Series) -> dict:
    cluster_id = int(row["cluster"])
    return {
        "name": row["Name"],
        "type1": row["Type 1"],
        "type2": row["Type 2"] if pd.notna(row["Type 2"]) else None,
        "hp": int(row["HP"]),
        "attack": int(row["Attack"]),
        "defense": int(row["Defense"]),
        "sp_atk": int(row["Sp. Atk"]),
        "sp_def": int(row["Sp. Def"]),
        "speed": int(row["Speed"]),
        "total": int(
            row[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]].sum()
        ),
        "cluster": cluster_id,
        "role": _cluster_roles[cluster_id]["role"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Pokemon Team Builder", lifespan=lifespan)


@app.get("/pokemon")
def list_pokemon():
    """List all pokemon with their cluster and role."""
    df = _model_data["df"]
    return [_pokemon_dict(row) for _, row in df.iterrows()]


@app.get("/pokemon/{name}")
def get_pokemon(name: str):
    """Get cluster info for a single pokemon by name."""
    df = _model_data["df"]
    matches = df[df["Name"].str.lower() == name.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Pokemon '{name}' not found.")
    return _pokemon_dict(matches.iloc[0])


@app.get("/team/suggest")
def suggest_team(starter: str):
    """
    Suggest a balanced 6-member team starting from a given pokemon.
    Picks the strongest representative from each of the remaining clusters.
    """
    df = _model_data["df"]
    n_clusters = _model_data["model"].n_clusters

    starter_matches = df[df["Name"].str.lower() == starter.lower()]
    if starter_matches.empty:
        raise HTTPException(status_code=404, detail=f"Pokemon '{starter}' not found.")

    starter_row = starter_matches.iloc[0]
    starter_cluster = int(starter_row["cluster"])

    team = [starter_row]
    other_clusters = [c for c in range(n_clusters) if c != starter_cluster]

    for cluster_id in other_clusters:
        if len(team) >= 6:
            break
        cluster_df = df[df["cluster"] == cluster_id].copy()
        cluster_df["_total"] = cluster_df[
            ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
        ].sum(axis=1)
        candidates = cluster_df.sort_values("_total", ascending=False)
        if not candidates.empty:
            team.append(candidates.iloc[0])

    roles_covered = list({_cluster_roles[int(p["cluster"])]["role"] for p in team})

    return {
        "team": [_pokemon_dict(p) for p in team],
        "roles_covered": sorted(roles_covered),
    }


@app.get("/clusters")
def get_clusters():
    """Show all cluster roles and their stats."""
    return {str(k): v for k, v in _cluster_roles.items()}


@app.post("/admin/reload-model")
def reload_model():
    """Reload model.pkl from disk without restarting the server."""
    _load_model()
    return {"status": "model reloaded"}
