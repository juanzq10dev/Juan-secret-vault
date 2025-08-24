= Clustering

*Definition:* Clustering is an unsupervised learning technique that partitions data points into clusters based on similarity patterns.

*Problem it solves: *
- Automatically finds hidden groups in data without labels.
- It reveals hidden patterns and similarities.

*How it works:*
- It uses K-mean algorithm.

== K-mean algorithm
*Definition:* It is the algorithm used to group data in $K$ clusters.

*How it works:*
+ Randomly initialize $K$ ($K = "number of clusters"$) points ($mu$ = cluster centroid), named cluster centroids, which are the center of a cluster.
+ Classifies each point depending on the closes centroid.
+ Computes average of each cluster and moves centroids to that point.
+ Repeat previous steps until convergence (no point classification changes, so centroids do not change too).

```
Initialize centroids mu randomly.
Repeat until convergence:
  for i = 1 to m
    # Assign points to cluster centroids
    c_i := index (from 1 to k) of cluster centroid closes to x^(i)

  for k = 1 to K
    # Move cluster centroids
    mu_k = average (mean) of points assigned to cluster k
```

== K-mean cost function
K-mean uses the distortion cost function:
$
  J(c^((i)), ..., c^((m)); mu_1, ..., mu_k) = 1 / m sum_(i = 1)^m || x^((i)) - mu_(c^((i)))||^2
$

Where:
- $m$: number of values in the dataset.
- $c_((i))$: index of cluster $(1, 2, k)$ to which example $x^((i))$ is assigned.
- $mu_k$: cluster centroid $k$

By definition K-mean algorithm never diverges, always reduces $J$.

== Random initialization.
*Definition:* It is a technique to initialize cluster centroids in an optimal way.

*Problem it solves:* Depending on the starting point of cluster centroids, K-mean may may be diverge faster or not.

*How it works:*
+ Given  $K < m$.
+ Randomly pick $K$ training examples.
+ Set each cluster centroid ($mu_1, mu_2, mu_..., mu_k$) equal to the picked training examples.
+ Calculate $J$.
+ Continue with the process a random defined value and pick the set of clusters with least $J$.

== Choosing a good number of clusters.
- It is actually very ambiguous.
- Evaluate number of clusters based on how well it performs on the problem you want to solve.
- You can use elbow method if you are not sure.

=== Elbow method
*Definition:* A technique to define the number of clusters.

*Problem it solves:* Sometimes you may not know how many clusters add. (But actually it is very ambiguous).

*How it works:*
+ For each number of clusters $K$, calculate $J$.
+ Where you see $J$ to stop decreasing, forms like an elbow. That is the point you are likely to choose.
