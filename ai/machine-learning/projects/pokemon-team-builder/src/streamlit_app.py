import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path("model.pkl")

_STAT_TO_ROLE = {
    "HP": "Tank",
    "Attack": "Physical Sweeper",
    "Defense": "Defensive Wall",
    "Sp. Atk": "Special Attacker",
    "Sp. Def": "Special Wall",
    "Speed": "Fast Attacker",
}

STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        st.error("model.pkl not found — run `python src/api/train.py` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]


def compute_cluster_roles(df: pd.DataFrame, n_clusters: int) -> dict[int, dict]:
    ratio_cols = [f"{col}_ratio" for col in STAT_COLS]
    use_ratios = all(col in df.columns for col in ratio_cols)
    roles = {}
    for cid in range(n_clusters):
        group = df[df["cluster"] == cid]
        if use_ratios:
            dominant = str(group[ratio_cols].mean().idxmax()).replace("_ratio", "")
        else:
            dominant = str(group[STAT_COLS].mean().idxmax())
        roles[cid] = {
            "role": _STAT_TO_ROLE[dominant],
            "dominant_stat": dominant,
            "avg_total": round(group[STAT_COLS].sum(axis=1).mean(), 1),
            "size": len(group),
        }
    return roles


def pokemon_row_to_dict(row: pd.Series, cluster_roles: dict[int, dict]) -> dict:
    cid = int(row["cluster"])
    return {
        "Name": row["Name"],
        "Type 1": row["Type 1"],
        "Type 2": row["Type 2"] if pd.notna(row["Type 2"]) else "—",
        "HP": int(row["HP"]),
        "Attack": int(row["Attack"]),
        "Defense": int(row["Defense"]),
        "Sp. Atk": int(row["Sp. Atk"]),
        "Sp. Def": int(row["Sp. Def"]),
        "Speed": int(row["Speed"]),
        "Total": int(row[STAT_COLS].sum()),
        "Cluster": cid,
        "Role": cluster_roles[cid]["role"],
    }


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pokemon Team Builder", page_icon="pokeball", layout="wide")

col_title, col_btn = st.columns([8, 1])
with col_title:
    st.title("Pokemon Team Builder")
with col_btn:
    st.write("")
    if st.button("Reload model"):
        load_model.clear()
        st.rerun()

data = load_model()
df: pd.DataFrame = data["df"]
n_clusters: int = data["model"].n_clusters
cluster_roles = compute_cluster_roles(df, n_clusters)

tab_team, tab_clusters, tab_lookup = st.tabs(
    ["Team Builder", "Cluster Explorer", "Pokemon Lookup"]
)

# ── Tab 1: Team Builder ───────────────────────────────────────────────────────

with tab_team:
    st.header("Build a Balanced Team")
    st.write("Pick a starter and get a balanced 6-member team covering all roles.")

    all_names = sorted(df["Name"].tolist())
    starter_name = st.selectbox("Choose your starter", all_names)

    if st.button("Suggest Team", type="primary"):
        starter_matches = df[df["Name"].str.lower() == starter_name.lower()]
        starter_row = starter_matches.iloc[0]
        starter_cluster = int(starter_row["cluster"])

        team_rows = [starter_row]
        for cid in range(n_clusters):
            if len(team_rows) >= 6:
                break
            if cid == starter_cluster:
                continue
            cluster_df = df[df["cluster"] == cid].copy()
            cluster_df["_total"] = cluster_df[STAT_COLS].sum(axis=1)
            best = cluster_df.sort_values("_total", ascending=False).iloc[0]
            team_rows.append(best)

        team = [pokemon_row_to_dict(r, cluster_roles) for r in team_rows]
        roles_covered = sorted({m["Role"] for m in team})

        st.subheader("Your Team")
        cols = st.columns(6)
        for col, member in zip(cols, team):
            with col:
                st.metric(member["Name"], member["Role"])
                st.caption(
                    f"Types: {member['Type 1']}"
                    + (f" / {member['Type 2']}" if member["Type 2"] != "—" else "")
                )
                st.caption(f"Total: {member['Total']}")

        st.subheader("Roles Covered")
        st.write(", ".join(roles_covered))

        st.subheader("Full Stats")
        team_df = pd.DataFrame(team).set_index("Name")
        st.dataframe(team_df, use_container_width=True)

# ── Tab 2: Cluster Explorer ───────────────────────────────────────────────────

with tab_clusters:
    st.header("Cluster Explorer")
    st.write("Browse all 6 clusters and the Pokemon that belong to each one.")

    for cid in range(n_clusters):
        role_info = cluster_roles[cid]
        with st.expander(
            f"Cluster {cid} — {role_info['role']}  "
            f"({role_info['size']} Pokemon, avg total {role_info['avg_total']})"
        ):
            cluster_df = df[df["cluster"] == cid][
                ["Name", "Type 1", "Type 2"] + STAT_COLS
            ].copy()
            cluster_df["Total"] = cluster_df[STAT_COLS].sum(axis=1)
            cluster_df = cluster_df.sort_values("Total", ascending=False).reset_index(
                drop=True
            )
            st.dataframe(cluster_df, use_container_width=True)

# ── Tab 3: Pokemon Lookup ─────────────────────────────────────────────────────

with tab_lookup:
    st.header("Pokemon Lookup")

    search = st.text_input("Search by name")

    if search:
        results = df[df["Name"].str.lower().str.contains(search.lower())]
        if results.empty:
            st.warning(f"No Pokemon found matching '{search}'.")
        else:
            for _, row in results.iterrows():
                info = pokemon_row_to_dict(row, cluster_roles)
                with st.expander(f"{info['Name']}  —  {info['Role']}"):
                    left, right = st.columns(2)
                    with left:
                        st.write(f"**Type:** {info['Type 1']}" + (f" / {info['Type 2']}" if info["Type 2"] != "—" else ""))
                        st.write(f"**Cluster:** {info['Cluster']}")
                        st.write(f"**Role:** {info['Role']}")
                    with right:
                        stat_data = {s: info[s] for s in STAT_COLS}
                        st.bar_chart(stat_data)
