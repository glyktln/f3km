# F3KM – Fairness-Aware K-Means Clustering

## Overview

This project implements **F3KM**, a fairness-aware extension of K-Means clustering.
The algorithm incorporates a **protected attribute (binary: 0/1)** into the clustering process, aiming to balance groups within each cluster while maintaining clustering quality.

## Key Idea

F3KM modifies standard K-Means by:

* Adjusting **assignment step** using fairness-aware forces
* Modifying **centroid updates** to reduce imbalance
* Introducing a trade-off parameter `lambda_` between **quality and fairness**

## Features

* Multiple initialization strategies:

  * `random_centroids`
  * `random_labels`
  * `random_c_kmeans`
* Multiple centroid update rules:

  * `standard`
  * `simple_weighted`
  * `shifted`
  * `weighted`
  * 
* Supports:
  * **Normalized (`norm`)** and **unnormalized (`unorm`)** balance

## Usage
```python
model = F3KMEANS(
    n_clusters=10,
    type_balance='norm',
    type_init='random_c_kmeans',
    type_update_centroids='shifted',
    lambda_=0.5
)

model.fit(X, attributes)
fairness = model.fairness_metrics()
quality = model.quality_metrics(X)
```

## Inputs

* `X`: feature matrix (numpy array)
* `attributes`: binary protected attribute (0/1)

## Outputs

* `labels_`: cluster assignments
* `centroids`: final cluster centers
* `fairness_metrics()`: fairness evaluation
* `quality_metrics()`: clustering quality

## Fairness Metrics

* `Balance`: average ratio of minority to majority group within clusters
* `N_Balance`: average normalized balance
* `FR`: worst-case cluster balance
* `FR_norm`: normalized worst-case bairness

## Quality Metrics

* `Silhouette`: cluster separation
* `SSE`: sum of squared errors

## Termination

The algorithm stops when:

* Centroids converge
* Oscillation is detected
* Maximum iterations reached



