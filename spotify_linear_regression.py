from networkx.algorithms import approximation as approx
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql.functions import col,isnan,when,count
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

spark = SparkSession.builder \
    .master("local[*]") \
    .getOrCreate()

import kagglehub

# Download latest version
path = kagglehub.dataset_download("jfreyberg/spotify-artist-feature-collaboration-network")

print("Path to dataset files:", path)

path_csv = "/root/.cache/kagglehub/datasets/jfreyberg/spotify-artist-feature-collaboration-network/versions/2"

edges_df = pd.read_csv(path_csv + '/edges.csv')
nodes_df = pd.read_csv(path_csv + '/nodes.csv')

edges_df.head()

nodes_df.head()

all_nodes = set(nodes_df["spotify_id"])

nodes_df.shape

edges_df.shape

"""# Create Graph using NetworkX"""

G = nx.from_pandas_edgelist(edges_df, source='id_0', target='id_1')

G.add_nodes_from(all_nodes)

node_list = list(G.nodes)

pandas_nodes = pd.DataFrame(G.degree(node_list), columns=['node', 'degree'])
pandas_nodes.head()

mean_degree = pandas_nodes["degree"].mean()
print("Mean degree:", mean_degree)

"""# Computing Centrality Measures"""

# Taking centrality measures of the whole graph is computationally expensive, we will be taking them of a subgraph
random_nodes = np.random.choice(list(G.nodes()), size=int(len(G.nodes()) / 2), replace=False)
H = G.subgraph(random_nodes)

id_to_name = dict(zip(nodes_df["spotify_id"], nodes_df["name"]))

degree_centrality = nx.degree_centrality(H)

betweenness = nx.betweenness_centrality(H, k=500)

closeness = nx.closeness_centrality(H)

eigenvector = nx.eigenvector_centrality(H, max_iter=100, tol=1e-03)

pagerank = nx.pagerank(H)

"""# Top 5-Artists per Centrality Measurement"""

def print_top_5(centrality_dict, title):
  top5 = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:5]
  for id, score in top5:
      artist_name = id_to_name.get(id, id)
      print(f"{artist_name}: {score:.4f}")

print_top_5(degree_centrality, "Degree Centrality")

print_top_5(betweenness, "Betweenness Centrality")

print_top_5(closeness, "Closeness Centrality")

print_top_5(eigenvector, "Eigenvector Centrality")

print_top_5(pagerank, "PageRank")

"""# Predicting Artist Popularity with PySpark"""

nodes = list(degree_centrality.keys())

pd_centrality_df = pd.DataFrame({
    "Node": nodes,
    "Artist": [id_to_name.get(n, "Unknown") for n in nodes],
    "Degree": [degree_centrality.get(n, 0) for n in nodes],
    "Betweenness": [betweenness.get(n, 0) for n in nodes],
    "Closeness": [closeness.get(n, 0) for n in nodes],
    "Eigenvector": [eigenvector.get(n, 0) for n in nodes],
    "PageRank": [pagerank.get(n, 0) for n in nodes]
})

# Save to centrality pdf to local Colab file system
pd_centrality_df.to_csv("pd_centrality_scores.csv", index=False)

df1 = spark.read.csv("pd_centrality_scores.csv", header=True, inferSchema=True)
df1 = df1.withColumnRenamed("Node", "spotify_id")
df1.show()

df2 = spark.read.csv("spotify_nodes.csv", header=True, inferSchema=True)
df2 = df2.drop("name")
df2.show()

df = df1.join(df2,['spotify_id'],how='inner')

# This spotify ID was causing issues when merged
df = df.filter(col("spotify_id") != "1hGJNACUxxr1vMX3HLimGP")

"""### Data Preprocessing"""

null = df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c
                           )).alias(c)
                    for c in df.columns])
null.show()

# Change the 'followers' column in place to integer
# Replaced NA with 0
df = df.withColumn("followers", col("followers").cast("int"))
df = df.fillna(0, subset=['followers'])
df.show()

# Change the 'Degree' column in place to integer
df = df.withColumn("Degree", col("Degree").cast("float"))

from pyspark.sql.functions import log1p, col
#Log‐transform the followers column
df = df.withColumn("log_followers", log1p(col("followers")))

featureCols=["Degree", "Betweenness", "Closeness", "Eigenvector", "log_followers"]

# put features into a feature vector column
assembler = VectorAssembler(inputCols=featureCols, outputCol="features", handleInvalid="skip")

assembled_df = assembler.transform(df)
assembled_df.show(10, truncate=False)

# Initialize the StandardScaler
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
# Fit the DataFrame to the scaler
scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)
# Inspect the result
scaled_df.select("features", "features_scaled").show(10, truncate=False)

# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2], seed=420)

# Initialize `lr` no Lasso
lr = (LinearRegression(featuresCol='features_scaled', labelCol="popularity", predictionCol="pred_popularity"))

# Fit the data to the model
linearModel = lr.fit(train_data)

# Coefficients for the model
linearModel.coefficients

coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]

coeff_df

# Generate predictions
predictions = linearModel.transform(test_data)

# Extract the predictions and the "known" correct labels
predandlabels = predictions.select("Artist","popularity", "pred_popularity")

predandlabels.show()

sorted_df = predandlabels.orderBy(col("Popularity").desc())
sorted_df.show()

"""# Evaluate Model Performance"""

# Get the RMSE
print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))

# Get the MAE
print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))

# Get the R2
print("R2: {0}".format(linearModel.summary.r2))

