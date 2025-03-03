#ks #kolmogorov-smirnov #binary_classification #data_distribution
# Kolmogorov-Smirnov (KS)

## What is the KS test?

The Kolmogorov-Smirnov (KS) test is a non-parametric test used to compare two distributions. It can be used to:

1. Compare a sample with a reference probability distribution (one-sample KS test).
2. Compare two samples (two-sample KS test)

## Why use the KS test in Machine Learning?


In machine learning, the KS test is often used to evaluate the performance of models, particularly in binary classification problems. It **helps measure how well the model separates the positive and negative classes**. 

The KS statistic quantifies the maximum distance between the Cumulative Distribution Functions ([ECDF](ECDF.md)) of the predicted probabilities for the two classes.

## Steps to Understand the KS Test with an Example

Let's go through a step-by-step example.
[Exercise calculating KS](./ks.py)

### Step 1: Prepare Your Data

Suppose you have a binary classification model, and you have predicted probabilities for the positive class and the actual labels for your test set.

```python
import numpy as np
import pandas as pd

# Example predicted probabilities and actual labels
predicted_probabilities = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.2, 0.3, 0.1, 0.5, 0.55])
actual_labels = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1])
```

### Step 2: Separate Predictions by Class

Separate the predicted probabilities into two groups based on the actual labels.
```python
positive_probabilities = predicted_probabilities[actual_labels == 1] negative_probabilities = predicted_probabilities[actual_labels == 0]
```

### Step 3: Calculate  the Empirical Distribution Function for each group

[ECDF](ECDF.md)

```python
def ecdf(data): 
	# Sort data 
	x = np.sort(data) 
	# Calculate the ECDF values 
	y = np.arange(1, len(data)+1) / len(data) 
	return x, y 
	
pos_x, pos_ecdf = ecdf(positive_probabilities) 
neg_x, neg_ecdf = ecdf(negative_probabilities)
```


### Step 4: Calculate the KS statistic

The KS statistic is the maximum distance between the ECDFs of the two groups.



