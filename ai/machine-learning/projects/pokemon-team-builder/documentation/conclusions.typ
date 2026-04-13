#set document(title: "Pokemon Team Builder — Clustering Experiments & Conclusions")
#set page(margin: 2cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#show link: underline

#align(center)[
  #text(size: 18pt, weight: "bold")[Pokemon Team Builder]
  #linebreak()
  #text(size: 13pt)[Clustering Experiments & Conclusions]
  #linebreak()
  #text(size: 10pt, fill: gray)[HeartGold / SoulSilver — Johto Pokedex]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

= Initial Setup

Started with `cluster.py`: KMeans on 6 raw stats + one-hot type encoding.
Dataset: `pokemon_hgss.csv` — Johto Pokedex for HeartGold/SoulSilver.

*Problem found:* CSV had 253 pokemon vs 270 on pokemondb.net.

*Fix:* Scraped pokemondb for the full list and missing stats (19 pokemon).
Added 14 extra obtainable pokemon (baby forms, Sinnoh evolutions) as \#257+.
Final dataset: *270 pokemon*.

= App Architecture

Converted the exploration script into a production-ready FastAPI app.

*Pattern used: Offline Batch*

```
train.py  → trains KMeans → saves model.pkl
main.py   → loads model.pkl once at startup → serves predictions
```

Key insight: `model.predict()` is microseconds. The model lives in memory.
Retraining is decoupled from serving via `POST /admin/reload-model`.

*Endpoints:*

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  [*Route*], [*Description*],
  [`GET /pokemon`], [List all pokemon with cluster and role],
  [`GET /pokemon/{name}`], [Cluster + role for one pokemon],
  [`GET /team/suggest`], [Suggest balanced 6-member team],
  [`GET /clusters`], [Cluster role summary],
  [`POST /admin/reload-model`], [Hot-reload model.pkl without restart],
)

= Feature Engineering

== Initial approach: Clustering by raw stat values

*Initial approach:* KMeans on raw stat values (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed).
Each pokemon represented as a 6-dimensional vector of absolute numbers. Total stat was excluded as could bias the model.

```python
stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
features = StandardScaler().fit_transform(df[stat_cols])
kmeans = KMeans(n_clusters=6).fit(features)
```

*Hypothesis:* Model would group Ookemons by its best stat, so, it would group by roles: Walls, Attacker, Sp. Attacker, etc.

*Result:* Weak/baby pokemon polluted a whole cluster. Pokemon with huge stats polluted another. It was not as expected.

== Fix 1: Adding Pokemon Type 

*Fix attempt:* Added type identity via one-hot encoding, and increased `TYPE_WEIGHT` to 3.5 so types would pull same-type pokemon together still considering the stats.

```python
type1_dummies = pd.get_dummies(df['Type 1'], prefix='type')
type2_dummies = pd.get_dummies(df['Type 2'].fillna('None'), prefix='type')
type_dummies = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)

TYPE_WEIGHT = 2.0
features = np.hstack([stats_norm, type_dummies.values * TYPE_WEIGHT])
```

*Result:* Clusters improved: pokemon grouped more coherently. But there where the following problems:
- Legendary Pokemon were moving the centroid of attackers, and grouping a huge part of the best Pokemons on a single cluster.
- Type weight was causing some Pokemons at the wrong cluster: example Tyranitar (Rock type) appeared with the Walls, despites being an offensive Pokemons because many walls are rock type. Lowering the TYPE WEIGHT  was causing the weak Pokemons cluster appear again.

== Fix 2: Base stats / Total stat

At this point I got back to use only the base stats, but this time dividing each stat by the pokemon's total status to remove power-level bias.

```python
df[f'{col}_ratio'] = df[col] / df['_total']
```

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  [*Pokemon*], [*HP*], [*HP Ratio*],
  [Cleffa],  [50],  [50/218 = 0.23],
  [Blissey], [255], [255/540 = 0.47],
)

*Result:* Weak pokemon distributed across role clusters by their actual proportions. But KMean random init was causing clusters with duplicate dominant stats. 

*Problem:* KMeans random init caused duplicate dominant stats.
Two clusters both ended up with Attack as dominant stat.

== Fix 3: Biased centroids

*Fix:* Custom `init=centroids` with one archetype per stat.

```python
centroids = np.array([
    [ 3,  0,  0,  0,  0,  0],  # Tank
    [ 0,  3,  0,  0,  0,  0],  # Physical Sweeper
    [ 0,  0,  3,  0,  0,  0],  # Defensive Wall
    [ 0,  0,  0,  3,  0,  0],  # Special Attacker
    [ 0,  0,  0,  0,  3,  0],  # Special Wall
    [ 0,  0,  0,  0,  0,  3],  # Fast Attacker
])
```

*Important:* `n_init=1` is required when using custom centroids.
Otherwise sklearn ignores them and runs random restarts.

== I gave up to Kingler — The Border Case

KMeans consistently placed Kingler in Defensive Wall regardless of:
- Centroid tuning (tried many combinations)
- `TYPE_WEIGHT` adjustments (down to 0)

*Stats:* HP=55 Atk=130 Def=115 SpA=50 SpD=50 Spe=75

*Ratios:*
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [HP], [*Atk*], [Def], [SpA], [SpD], [Spe],
  [0.12], [*0.27*], [0.24], [0.11], [0.11], [0.16],
)

Attack (0.27) and Defense (0.24) are very close.

*Root cause:* After convergence, the Defensive Wall centroid drifted to a region closer to Kingler than Physical Sweeper's centroid.

#table(
  columns: (1fr, auto),
  stroke: 0.5pt,
  [*Cluster*], [*Distance*],
  [Defensive Wall], [1.97 ← assigned],
  [Physical Sweeper], [3.75],
)

At this point, I was close to gave up, I realized I was thinking wrong:
+ If I wanted to classify Pokemons by its highest stat, I could just do it programmatically and did not needed machine learning for that. 
+ I had predefined clusters, I mean, I wanted groups considering high HP, Attack, Sp. Attack. This is actually a labeled data problem.

== Another approach: Weakness-Based Clustering

Replaced stats with a 17-value weakness profile as KMeans features.
No `StandardScaler` needed: multipliers (0, 0.5, 1, 2, 4) are already comparable.

*Result:* Clusters grouped by shared type weaknesses, not roles.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  [*Cluster*], [*Weak to*], [*Resists*],
  [0 — Normal], [Fighting 1.7x], [Ghost, Electric],
  [1 — Rock/Ground], [Water 4x, Grass 4x], [Electric (immune)],
  [2 — Water/Ghost], [Grass 2x, Electric 1.7x], [Fire, Ice],
  [3 — Grass], [Fire 2.2x, Flying 2.1x], [Grass, Fighting],
  [4 — Flying], [Rock 2.5x, Ice 2.1x], [Ground (immune)],
  [5 — Fire], [Ground 2.1x, Water 1.5x], [Grass, Steel],
)

*Team test for Meganium (Grass):*
Suggested: Tyranitar, Golem, Mewtwo, Lugia, Entei.
Remaining gap: Water (10.5x total exposure across team).

= The Solution: Combined Features

Combined all three feature sets:

```python
features = np.hstack([
    stats_norm,                                  # 6 cols  — role
    type_dummies.values * TYPE_WEIGHT,           # 17 cols — type identity
    weakness_matrix * WEAKNESS_WEIGHT,           # 17 cols — defensive profile
])
```

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt,
  [*Feature group*], [*Cols*], [*Purpose*],
  [Stat ratios], [6], [Drives role assignment],
  [Type dummies], [17], [Groups by type family],
  [Weakness matrix], [17], [Separates by what threatens them],
)

*Current working values:* `TYPE_WEIGHT=1.0`, `WEAKNESS_WEIGHT=1.5`

And after moving a bit the values. It worked