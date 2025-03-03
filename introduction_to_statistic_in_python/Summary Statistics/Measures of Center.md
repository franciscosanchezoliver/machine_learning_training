
In this chapter, we'll look at data about different mammals sleep habits.

![[Pasted image 20241027081051.png]]

Before we dive in, let's remind ourselves how histograms work. A histogram take a bunch of data points and separates them into bins, or ranges of values.

Here, there's a bin for 0 to 2 hours, 2 to 4 hours, and so on.

![[Pasted image 20241027081958.png]]
The heights of the bars represents the number of data points that fall into that bin, so there's one mammal in the dataset that sleeps between 0 and 2 hours, and nine mamnals that sleep two to four hours.

Histograms are a great way to visually summarize the data, but we can use numeric summary statistics to summarize even further.

One way we could summarize the data is by answering the question, How long do mammals in this dataset typically sleep? 

To answer this, we need to figure out **what the "typical" or "center" value of the data** is.

We'll discuss three different definitions, or measures of center: mean, median, and mode.

## Mean

The mean, often called the average, is one of the most common ways of summarizing data.

![[Pasted image 20241031081144.png]]

To calculate mean, we add up all the numbers of interest and divide by the total number of data points.

$$ 12.1 + 17.0 + 14.4 + 14.9 + 4.0 = 12.48 $$

In python we can use numpy's mean function, passing it the variable of interest.

```python
import numpy as np
np.mean(msleep["sleep_total"])
```

## Median

Another measure of center is the median.

The median is the value where 50% of the data is lower than it, and 50% of the data is higher. We can calculate this by sorting all the data points and taking the middle one (which in this case is index 41 with the value of 10.1)

```python
msleep['sleep_total'].sort_values().iloc[41]
```

This gives us a median of 10.1 hours of sleep.

In python we can use the np package to calculate the median:
```python
np.median(mlsleep['sleep_total'])
```


## Mode

The mode is the most frequent value in the data. 

If we count how many occurrences there are of each sleep and sort in descending order, there are 4 mammals that sleep 12.5 hours, so this is the mode.

```python
msleep["sleep_total"].value_counts()
```

![[Pasted image 20241102084856.png]]

The mode of the "vore" variable  (herbivore, onmivore, insectivore) is herbivore.
```python
msleep["vore"].value_counts()
```

![[Pasted image 20241102085342.png]]

We can also find the mode using the mode function from the statistics module.

```python
import statistics
statistics.mode(mlsleep["vore"])
```

Mode is often used for categorical variables, since categorical variables can be unordered and often don't have a numerical representation.

Now that we have a lot of ways of measure center, how do we know which one to use? Let's look at an example.

Here we have all the insectivores in the dataset:
```python
msleep[msleep["vore"] == "insecti"]
```

![[Pasted image 20241102090008.png]]

Here, we get a mean sleep time of 16.5 hours and a median sleep time of 18.9 hours.

```python
msleep[msleep["vore"]] == "insecti"]["sleep_total"].agg([np.mean, np.median])
```

![[Pasted image 20241103075703.png]]

Now, let's say that we have discovered a new mystery insectivore that never sleeps (sleep to 0). 
![[Pasted image 20241103075831.png]]

If we get the mean and median again, we get different results. The mean went down more than 3 hours, while the median changed by less than an hour.

![[Pasted image 20241103080236.png]]

This is because the mean is much more sensitive to extreme values than the median.

Since the mean is more sensitive to extreme values, it works better for symmetrical data like this:

![[Pasted image 20241103080445.png]]

```python
import matplotlib.pyplot as plt

# Histogram
data["values"].hist()
plt.show()
```

Notice that the mean (in read) and the median (in black), are quite close.

However, if the data is skewed, meaning it's not symmetrical, like this, median is usually better to use.

![[Pasted image 20241103080830.png]]

In this histogram, the data is piled up on the right, with a tail on the left. Data that looks like this is called left-skewed data.

When data is piled up on the left with a tail on the right, it's right-skewed.
![[Pasted image 20241103081542.png]]


When data is skewed, the mean and median are different. 

The mean is pulled in the direction of the skew, so it's lower than the median on the left-skewed data, and higher on the right skewed data.

![[Pasted image 20241103082655.png]]


Because the mean is pulled around by the extreme values, it's better to use the median since it's less affected by outliers.

