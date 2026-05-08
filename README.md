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








# Fair Clustering & Classification Data Preprocessing

This repository contains data preprocessing and sampling scripts for several widely used datasets in machine learning. The primary goal is to prepare these datasets for **Fairness in Clustering and Classification** research by isolating numerical features and extracting protected/sensitive attributes into separate labels.

## 🛠 Preprocessing Pipeline

For each dataset, the preprocessing script performs the following operations:
1. **Data Cleaning:** Removes invalid rows, missing values, or instances with "unknown" entries.
2. **Feature Extraction:** Isolates continuous/numerical attributes to be used by the learning algorithms (saved as `[dataset_name].txt`). Categorical variables are either encoded or dropped depending on the dataset.
3. **Protected Attribute Isolation:** Extracts the sensitive attribute (e.g., Gender, Marital Status) and encodes it as a binary variable (saved as `[dataset_name]_Color.txt`). This ensures the algorithm does not train directly on the sensitive attribute, allowing it to be used purely for fairness evaluation (e.g., calculating Balance or Disparate Impact).

## 📊 Datasets 

Widely used datasets for evaluating fairness in clustering and classification tasks. The metrics below reflect the **actual cleaned data** produced by our preprocessing scripts.

| Dataset | Cleaned Samples | Features | Protected Attribute | Task / Description | Source |
|---------|-----------------|----------|---------------------|--------------------|--------|
| **Adult (Census Income)** | 32,561 | 6 | Gender (Sex) | Predict whether income > $50K/year based on demographic and employment info. | [Link](https://archive.ics.uci.edu/dataset/2/adult) |
| **Bank Marketing** | 41,108 | 9 | Marital Status | Predict if a client subscribes to a term deposit from Portuguese bank campaigns. | [Link](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Credit Card Default** | 30,000 | 14 | Gender (Sex) | Predict probability of credit card default based on demographic and repayment history. | [Link](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Census (US 1990)** | 2,458,285 | 67 | Gender (Sex) | Socio-economic dataset from US Census Bureau, widely used as a benchmark for fair clustering. | [Link](https://archive.ics.uci.edu/dataset/116/us+census+data+1990) |
| **Diabetes (Hospital)** | 101,000 | 47 | Race, Gender, Age | Predict 30-day readmission from 10 years of US hospital diabetic patient records. | [Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **ACSIncome** | 1,660,000 | 10 | Gender, Race | Modern alternative to Adult dataset (2018). Predict income with flexible thresholds. | [Link](https://www.openml.org/search?type=data&sort=runs&id=43141&status=active) |

*Note: The Feature count represents the specific continuous/numerical variables extracted during this project's preprocessing phase, not the raw column count of the original files.*

## Data Reduction & Stratified Sampling Strategy

To optimize computational efficiency and model training times, a **Stratified Sampling** technique was applied across the preprocessed datasets. 

* **Standard Datasets (Adult, Bank, Credit Card):** Retained exactly **10%** of the original data volume.
* **Massive Datasets (US Census 1990):** Retained **2%** of the original data volume.

This approach was strictly chosen over simple random sampling to ensure **Data Integrity** and protect against class imbalance. By grouping the data by the protected label attribute before sampling, we guarantee that the smaller datasets perfectly mirror the statistical distribution and underlying patterns of the original, massive files. 

This results in highly representative subsets (`_sampled.txt` and `_Color_sampled.txt`) that drastically accelerate the algorithm testing phase without compromising the predictive validity or the fairness evaluation of the final machine learning models.



