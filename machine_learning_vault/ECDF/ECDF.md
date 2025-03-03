#ecdf #data_distribution 

# ECDF (Empirical Cumulative Distribution Function)

It's a way to visualize the distribution of a sample of data. It shows the 
proportion of observations less than or equal to a particular value in the
dataset.

## Why use ECDF?

ECDFs are usefull because they provide a complete view of the data distribution.
Unlike histograms, which depends on bin sizes and can be less precise, ECDFs are
non-parametric and don't rely on assumptions about the data distribution.

## How to construct an ECDF?

Let's go through the process with a simple example.

### Step 1: Sort Data

Start with a dataset. Let's take a small sample of data for simplicity:

```
Data: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
```

Sort the data:
```
Sorted Data: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

### Step 2: Calculate Proportions

Next, calculate the cummulative proportion of each data point. This means 
determining the proportion of data points less than or equal to each value 
in the sorted list.

1. For the first value (1), there are 2 occurrences of 1 out of 11 data points:

Sorted Data: [**1, 1**, 2, 3, 3, 4, 5, 5, 5, 6, 9]

```
ECDF(1) = 2/11 = 0.18
```

2. For the next value (2), there are 3 occurrencies of values <= 2

Sorted Data: [**1, 1, 2**, 3, 3, 4, 5, 5, 5, 6, 9]

```
ECDF(2) = 3/11 = 0.27
```

3. For the next value (3), there are 

Sorted Data: [**1, 1, 2, 3, 3**, 4, 5, 5, 5, 6, 9]

```
ECDF(2) = 5/11 = 0.45
```

### Plot ECDF

To visualize, plot the cumulative proportions against the sorted data values:

| Data Value | ECDF Value |
|:----------:|:----------:|
| 1          | 0.18       |
| 2          | 0.27       |
| 3          | 0.45       |
| 4          | 0.55       |
| 5          | 0.82       |
| 6          | 0.91       |
| 9          | 1.00       |


In a plot, the ECDF is a step function that jumps up at each data point. Here's
a Python code example to illustrate this:

[Python example](./ecdf.py)

Result:
![](Pasted%20image%2020240717075646.png)



