# Pokemon HG/SS Team Builder

This is Pokemon HG/SS Team Builder, a project to build your Pokemon Team for Pokemon HeartGold and Soul Silver applying clustering to build a diverse.

## How to run

**1. Create and activate the virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -e .
```

**3. Train the model**

Reads `data/pokemon_hgss.csv`, trains KMeans clustering, and saves `model.pkl`.

```bash
python src/train.py
```

**4. Start the API**

```bash
python -m uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/pokemon` | List all 253 pokemon with their cluster and role |
| `GET` | `/pokemon/{name}` | Get stats, cluster, and role for one pokemon |
| `GET` | `/team/suggest?starter={name}` | Suggest a balanced 6-member team |
| `GET` | `/clusters` | Show all 6 cluster roles and their average stats |

**Example:**

```bash
curl http://localhost:8000/pokemon/Chikorita
curl "http://localhost:8000/team/suggest?starter=Chikorita"
```
