import pickle
from pathlib import Path

import streamlit as st

from src.repositories.dataset_repository import PokemonColumn
from src.services.cluster_service import ClusterService
from src.services.team_builder_service import (
    ClassifiedPokemon,
    Team,
    TeamBuilderService,
)


MODEL_PATH = Path("model.pkl")

STAT_COLS = [
    PokemonColumn.HP,
    PokemonColumn.ATTACK,
    PokemonColumn.DEFENSE,
    PokemonColumn.SP_ATTACK,
    PokemonColumn.SP_DEFENSE,
    PokemonColumn.SPEED,
]


# ── Setup ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_services() -> tuple[TeamBuilderService, dict]:
    if not MODEL_PATH.exists():
        st.error("model.pkl not found — run `python -m src.training.train` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    cluster_service = ClusterService()
    team_service = TeamBuilderService(cluster_service)
    return team_service, data


team_service, model_data = load_services()
df = model_data["df"]
n_clusters: int = model_data["model"].n_clusters
cluster_roles = ClusterService().compute_roles(df, n_clusters)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pokemon Team Builder", layout="wide")

col_title, col_btn = st.columns([8, 1])
with col_title:
    st.title("Pokemon Team Builder")
with col_btn:
    st.write("")
    if st.button("Reload model"):
        load_services.clear()
        st.rerun()

tab_team, tab_clusters, tab_lookup = st.tabs(
    ["Team Builder", "Cluster Explorer", "Pokemon Lookup"]
)


# ── Tab 1: Team Builder ───────────────────────────────────────────────────────


def _render_team_member(col, member: ClassifiedPokemon) -> None:
    with col:
        st.metric(member.pokemon.name, member.role)
        type_label = member.pokemon.type1.value
        if member.pokemon.type2:
            type_label += f" / {member.pokemon.type2.value}"
        st.caption(f"Types: {type_label}")
        st.caption(f"Total: {member.pokemon.stats.total}")


with tab_team:
    st.header("Build a Balanced Team")
    st.write("Pick a starter and get a balanced 6-member team covering all roles.")

    all_names = sorted(df[PokemonColumn.NAME.value].tolist())
    starter_name = st.selectbox("Choose your starter", all_names)

    if st.button("Suggest Team", type="primary"):
        try:
            team: Team = team_service.suggest_team(df, n_clusters, starter_name)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.subheader("Your Team")
        cols = st.columns(6)
        for col, member in zip(cols, team.members):
            _render_team_member(col, member)

        st.subheader("Roles Covered")
        st.write(", ".join(team.roles_covered))

        st.subheader("Full Stats")
        team_data = [
            {
                "Name": m.pokemon.name,
                "Role": m.role,
                "Type 1": m.pokemon.type1.value,
                "Type 2": m.pokemon.type2.value if m.pokemon.type2 else "—",
                "HP": m.pokemon.stats.hp,
                "Attack": m.pokemon.stats.attack,
                "Defense": m.pokemon.stats.defense,
                "Sp. Atk": m.pokemon.stats.sp_attack,
                "Sp. Def": m.pokemon.stats.sp_defense,
                "Speed": m.pokemon.stats.speed,
                "Total": m.pokemon.stats.total,
            }
            for m in team.members
        ]
        st.dataframe(team_data, width="stretch")


# ── Tab 2: Cluster Explorer ───────────────────────────────────────────────────

with tab_clusters:
    st.header("Cluster Explorer")
    st.write("Browse all clusters and the Pokemon that belong to each one.")

    for cid, role_info in cluster_roles.items():
        with st.expander(
            f"Cluster {cid} — {role_info.role} "
            f"({role_info.size} Pokemon, avg total {role_info.avg_total})"
        ):
            stat_values = [col.value for col in STAT_COLS]
            cluster_df = df[df["cluster"] == cid][
                [
                    PokemonColumn.NAME.value,
                    PokemonColumn.TYPE1.value,
                    PokemonColumn.TYPE2.value,
                ]
                + stat_values
            ].copy()
            cluster_df["Total"] = cluster_df[stat_values].sum(axis=1)
            cluster_df = cluster_df.sort_values("Total", ascending=False).reset_index(
                drop=True
            )
            st.dataframe(cluster_df, width="stretch")


# ── Tab 3: Pokemon Lookup ─────────────────────────────────────────────────────

with tab_lookup:
    st.header("Pokemon Lookup")
    search = st.text_input("Search by name")

    if search:
        results = df[
            df[PokemonColumn.NAME.value].str.lower().str.contains(search.lower())
        ]
        if results.empty:
            st.warning(f"No Pokemon found matching '{search}'.")
        else:
            for _, row in results.iterrows():
                classified = team_service.to_classified_pokemon(row, cluster_roles)
                with st.expander(f"{classified.pokemon.name}  —  {classified.role}"):
                    left, right = st.columns(2)
                    with left:
                        type_label = classified.pokemon.type1.value
                        if classified.pokemon.type2:
                            type_label += f" / {classified.pokemon.type2.value}"
                        st.write(f"**Type:** {type_label}")
                        st.write(f"**Cluster:** {classified.cluster}")
                        st.write(f"**Role:** {classified.role}")
                    with right:
                        stat_data = {
                            col.value: getattr(classified.pokemon.stats, col.value)
                            for col in STAT_COLS
                        }
                        st.bar_chart(stat_data)
