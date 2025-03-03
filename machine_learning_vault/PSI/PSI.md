#PSI #stability #data_distribution
# PSI (Population Stability Index)

## What is PSI in machine learning

It's a metric used to **measure the stability of a variable's distribution** 
over time.

PSI is often **used** in the context of **monitoring the performance of predictive** 
**models**, particularly in finance, to **check if the distribution of input features** 
**or model scores has shifted significantly from a reference period** (like a
**training dataset**) **to a current period** (like **new incoming data**).

## Why is PSI important?

1. **Model Monitoring**: helps **detect if the model's input data distribution has**
**changed**, which could indicate that **the model might not perform as well as** it
did during **training**.

2. **Data Quality**: Helps **ensure the consistency of data over time**.

3. **Regulatory Requirements**: In some industries, **monitoring data stability** is
a regulatory requirements.

## How to calculate PSI?

Let's go through the steps to calculate PSI with a simple example.

### 1. Divide the Data into Bins
**Split** both **the reference and current data into bins**. These bins can be equal
width or based on quantiles.

### 2. Calculate the Distribution
**Calculate the percentage of observations in each bin for both reference and** 
**current data.**

### 3. Compute PSI for each bin
**Use** the **PSI formula for each bin** to measure the stability

$$
PSI = \sum ( (Ref\_Dist-Cur\_dist) · \ln{(\dfrac{Ref\_dist}{Cur\_dist}) )}
$$

### 4. Sum the PSI values
**Sum the PSI values for all bins** to get the overall PSI

### Example Calculation
Let's say we have the following reference and current data distributions:

Reference Data (Training Data): [100, 200, 150, 50]
Current Data (New Data): [80, 220, 140, 60]

Assume we **divide both datasets into 4 bins**:

1. **Calculate the percentage of observations in each bins**:

Reference Distribution: [0.2, 0.4, 0.3, 0.1]
Current Distribution: [0.16, 0.44, 0.28, 0.12]

2. **Apply the PSI formula for each bin**:

- Bin 1:
Ref_Dist = 0.2
Cur_dist = 0.16

$$
PSI = \sum ( (Ref\_Dist-Cur\_dist) · \ln{(\dfrac{Ref\_dist}{Cur\_dist}) )}
$$

$$
PSI_{bin1} = (0.2-0.16)·ln(\frac{0.2}{0.16})
$$

$$
PSI_{bin1} = 0.04 · ln(1.25)
$$

$$
PSI_{bin1} = 0.00892
$$

- Bin 2:

Ref_Dist = 0.4
Cur_dist = 0.44

	
$$
PSI = \sum ( (Ref\_Dist-Cur\_dist) · \ln{(\dfrac{Ref\_dist}{Cur\_dist}) )}
$$

$$
PSI_{bin2} = (0.4-0.44)·ln(\frac{0.4}{0.44})
$$

$$
PSI_{bin2} = -0.04 · ln(0.9090)
$$

$$
PSI_{bin2} = 0.00376
$$

- Bin 3:

Ref_Dist = 0.3
Cur_dist = 0.28

$$
PSI = \sum ( (Ref\_Dist-Cur\_dist) · \ln{(\dfrac{Ref\_dist}{Cur\_dist}) )}
$$

$$
PSI_{bin1} = (0.3-0.28)·ln(\frac{0.3}{0.28})
$$

$$
PSI_{bin3} = 0.00136
$$


- Bin 4:
$$
PSI_{bin4} = 0.00372
$$


Sum the PSI values:
$$
PSI = PSI_{bin1} + PSI_{bin2} + PSI_{bin3} + PSI_{bin4}
$$

$$
PSI = 0.00892 + 0.00376 + 0.00136 + 0.00372
$$
$$
 PSI = 0,01776
$$
### Interpretation

if **PSI < 0.1** -> Minor Shift, the **population is stable**.
if **0.1 <= PSI < 0.2** **Moderate Shift**, need to **monitor**.
if **PSI >= 0.2** **Major Shift**, **investigate the cause**.

In our example, the PSI of 0.01776 indicates a minor shift, suggesting that the population is stable.

### Summary

- **PSI is a tool to ensure data stability over time**
- It **involves comparing distributions of reference and current data** using bins.
- PSI **helps in monitoring model performance and ensuring data consistency**.



[Exercise](PSI.py)



