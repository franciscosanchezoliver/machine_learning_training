In this chapter we'll talk about another set of summary statistics, measures of spread. 

Spread is just what it sounds like, it describes how spread apart or close together the data points are. 

![[Pasted image 20241107080942.png]]

Just like measures of center, there are a few different measures of spread. 

## Variance

The first measure, variance, measures the average distance from each data point to the data's mean.

![[Pasted image 20241109102839.png]]

To calculate variance, we start by calculating the distance between each data point and the mean, so we get one number for every data point.

```python
dists = msleep['sleep_total'] - np.mean(msleep['sleep_total'])
print(dists)
```

![[Pasted image 20241109103332.png]]

We then square each distance. 

```python
sq_dists = dists ** 2
print(sq_dists)
```

![[Pasted image 20241109103620.png]]

And then, add them all together. 

```python
sum_sq_dists = np.sum(sq_dists)
print(sum_sq_dists)
```

![[Pasted image 20241109103943.png]]

And finally, we divide the sum of squared distances by the number of data points minus 1, giving us the variance. 

```python 
variance = sum_sq_dists / (83 - 1)
print(variance)
```

So the variance is: 
![[Pasted image 20241109104549.png]]

The higher the variance, the more spread out the data is. 

It's important to note that the units of variance are squared, so in this case, it's 19.8 hours squared. 

We can calculate the variance using the library numpy:
```python
np.var(msleep['sleep_total'], ddof=1)
```

![[Pasted image 20241109105010.png]]

Note: we have to set the parameter  "ddof" to 1. If we dont specify the ddof equals 1, a slightty different formula is used to calculate variance that should only be used on a full population, not a sample. 

## Standard Deviation

The standard deviation is another measure of spread, calculated by taking the squared root of the variance. 

It can be calculated using:
```python
np.sqrt(np.var(msleep['sleep_total'], ddof=1))
```

![[Pasted image 20241109110424.png]]

Or even easier as:
```python
np.std(msleep['sleep_total'], ddof=1)
```

![[Pasted image 20241109110424.png]]


The nice thing about standard deviation is that the units are usually easy to understand since the're not squared. 

It's easier to wrap your head around 4 and a half hours than 19.8 hours squared.

## Mean Absolute Deviation

Mean Absolute Deviation takes the absolute value of the distances to the mean, and then takes the mean of those diffences:
```python 
dists = msleep['sleep_total'] - mean(msleep$sleep_total)
np.mean(np.abs(dists))
```

![[Pasted image 20241111203340.png]]

While this is similar to the standard deviation, it's not exactly the same.

## Standard Deviation VS Mean Absolute Deviation

Standard deviation squares distances, penalizing longer distances more than shorter ones.

Mean Absolute Deviation penalizes each distance equally.

One isn't better than the other, but SD is more common than MAD


Before we discuss the next measure of spread, let's quickly talk about quantiles.

## Quantiles

Quantiles, also called percentiles, split up the data into some number of equal parts. 

Here, we call the function "quantile" passing the column of interest followed by 0.5:

```python
np.quantile(msleep['sleep_total'], 0.5)
```

![[Pasted image 20241111204120.png]]

This gives us 10.1 hours, so 50% of mammals in the dataset sleep less than 10.1 hours a day, and the other 50% sleep more than 10.1 hours, so this is exactly the same as the median.

We can also pass a list of numbers to get multiple quantiles at once.

```python
np.quantile(msleep['sleep_total'], [0, 0.25, 0.5, 0.75, 1])
```

![[Pasted image 20241111205643.png]]

Here, we split the data into 4 equal parts, these are also called "quartiles". 

This means that 25% of the data is between 1.9 and 7.85, another 25% is between 7.85 and 10.1, and so on.

The boxes in box plots represents quartiles.

```python
import matplotlib.pyplot as plt

plt.boxplot(msleep['sleep_total'])
plt.show()
```

![[Pasted image 20241111210654.png]]


The bottom of the box is the first quartile, and the top of the box is the third quartile. 

The middle line is the second quartile, the median. 

Here, we split the data into 5 equal pieces:
```python
np.quantile(msleep['sleep_total'], [0, 0.2, 0.4, 0.6, 0.8, 1])
```

![[Pasted image 20241111211721.png]]

But we can also use the function "np.linspace" as a shortcut, which takes in the starting number, the stopping number, and the number of intervals. 

