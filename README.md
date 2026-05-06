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
  * `weighted`
  * `simple_weighted`
  * `shifted`

* Supports:
  * **Normalized (`norm`)** and **unnormalized (`unorm`)** balance

## Usage
```python
model = F3KM(
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


---

## Datasets 

Widely used datasets for evaluating fairness in clustering and classification tasks. Each dataset includes sensitive attributes, making them benchmarks for bias and fairness research.

| Dataset        | Samples   | Features | Protected Attribute(s) | Task / Description | Source |
|----------------|-----------|----------|------------------------|--------------------|--------|
| Adult (Census Income) | 48K       | 14       | Gender, Race            | Predict whether income > $50K/year based on demographic and employment info. | [Link](https://archive.ics.uci.edu/dataset/2/adult) |
| Bank Marketing | 45K       | 16       | Gender, Marital status  | Predict if a client subscribes to a term deposit from Portuguese bank campaigns. | [Link](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Credit Card Default | 30K       | 23       | Gender, Education, Marital status | Predict probability of credit card default based on demographic and repayment history. | [Link](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Diabetes (Hospital) | 101K       | 47      | Race, Gender, Age       | Predict 30-day readmission from 10 years of US hospital diabetic patient records. | [Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| Census (US 1990) | 2.5M+     | 68       | Gender, Race, Marital status | Socio-economic dataset from US Census Bureau, widely used for fairness in clustering. | [Link](https://archive.ics.uci.edu/dataset/116/us+census+data+1990) |
| ACSIncome      | 1.66M     | 10      | Gender, Race            | Modern alternative to Adult dataset (2018). Predict income with flexible thresholds. | [Link](https://www.openml.org/search?type=data&sort=runs&id=43141&status=active) |



