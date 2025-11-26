# Predicting Spotify Artist Popularity from Network Centrality

This repository contains a PySpark / NetworkX project that predicts **Spotify artist popularity** using **linear regression** on graph-based **centrality measures** computed from an artist collaboration network.

The core idea:  
Artists are nodes in a collaboration graph; edges connect artists who have collaborated. We compute several **node centrality metrics** and use them, along with follower counts, to predict each artist’s Spotify popularity score.

---

## Project Overview

**Data source**

- Dataset: *Spotify Artist Feature & Collaboration Network* (Kaggle)  
- Downloaded programmatically using `kagglehub`  
- Key files from the dataset:
  - `edges.csv` — collaboration edges (artist \`spotify_id\` pairs)
  - `nodes.csv` — artist metadata (ID, name, followers, popularity, etc.)

**Main steps in the pipeline**

1. **Build the collaboration graph**
   - Use `NetworkX` to create an undirected graph from `edges.csv`.
   - Ensure all artists in `nodes.csv` exist as nodes in the graph.

2. **Compute centrality measures** (on a sampled subgraph for efficiency)
   - Degree centrality  
   - Betweenness centrality  
   - Closeness centrality  
   - Eigenvector centrality  
   - PageRank  
   - Map \`spotify_id\` → artist name for readability.

3. **Inspect most central artists**
   - Print the top 5 artists by each centrality metric to see who dominates the network structure.

4. **Prepare a modeling dataset (Pandas → CSV → Spark)**
   - Build a Pandas dataframe with:
     - \`spotify_id\`, \`Artist\`
     - \`Degree\`, \`Betweenness\`, \`Closeness\`, \`Eigenvector\`, \`PageRank\`
   - Save as \`pd_centrality_scores.csv\`.
   - Load it into Spark as \`df1\`.
   - Load artist metadata (including \`followers\`, \`popularity\`) as \`df2\` from \`spotify_nodes.csv\`.
   - Inner-join on \`spotify_id\` and drop problematic rows / NAs.
   - Log-transform followers with \`log_followers = log1p(followers)\`.

5. **Feature engineering & scaling (PySpark)**
   - Features used in the linear regression:
     - \`Degree\`
     - \`Betweenness\`
     - \`Closeness\`
     - \`Eigenvector\`
     - \`log_followers\`
   - Assemble these into a single feature vector using \`VectorAssembler\`.
   - Standardize features via \`StandardScaler\` to get \`features_scaled\`.

6. **Linear regression model**
   - Split into train/test: 80% / 20%.
   - Train a \`LinearRegression\` model with:
     - \`featuresCol="features_scaled"\`
     - \`labelCol="popularity"\`
     - \`predictionCol="pred_popularity"\`
   - Extract coefficients and intercept into a small Pandas table for inspection.
   - Generate predictions with \`pred_popularity\` for each artist in the test set.

7. **Model evaluation**
   - Report:
     - Root Mean Squared Error (**RMSE**)
     - Mean Absolute Error (**MAE**)
     - Coefficient of determination (**R²**)

---

## Model & Features

**Target variable**

- \`popularity\` — Spotify’s artist popularity score from the metadata.

**Features (final model)**

- \`Degree\` — Local connectivity / number of collaborations in the sampled graph.
- \`Betweenness\` — How often an artist lies on shortest paths between others.
- \`Closeness\` — Inverse average distance to all reachable nodes.
- \`Eigenvector\` — Influence of an artist by being connected to other influential artists.
- \`log_followers\` — Log-transformed follower count (stabilizes skew).

**Additional metric**

- \`PageRank\` is computed and available in the centrality CSV but not included in the current linear model. It’s a natural candidate for future experiments.

---

## Requirements

The core Python stack used:

```text
networkx
pyspark
kagglehub
pandas
numpy
matplotlib
seaborn
```

---
