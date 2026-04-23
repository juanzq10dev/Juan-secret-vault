# Pokemon HG/SS Team Builder

Suggests a balanced 6-member Pokemon team using KMeans clustering on Gen IV (HeartGold/SoulSilver) data.


## Getting Started

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

```bash
python -m src.training.train
```

**4. Run the app**

```bash
streamlit run src/ui/streamlit_app.py
```

## Conclusion
Clustering is powerful to find different patterns in data. The hardest part is to play with the data, I spent a huge time modifying the centroids, and playing with different weights to achieve the expected result.

- Kmean classifies data based on how far from centroids is.
- It is a good approach to find unknown data patterns.
- Modifying centroids and playing with weights is key to achieve your expected results.

## Note

This project has weak architecture, main idea was to produce a K-mean real implementation and that is achieved. So I see no point in continue developing other things that focus lean towards software development refinement and not too much about machine learning as this is an experimental project to practice machine learning.

Even documentation is a stale.
