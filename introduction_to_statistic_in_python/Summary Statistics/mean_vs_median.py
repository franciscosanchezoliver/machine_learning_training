# Mean vs. median
# ---------------
# The mean is the sum of all the data points divided by the total
# number of data points.
# The median is the middle value of the dataset where 50% of the
# data is less than the median, and 50% of the data is greater than
# the median.
# In this exercise, we'll compare these two measures of
# center.
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

food_consumption = pd.read_csv("data/food_consumption.csv")

# Subset for food_category equals rice
rice_consumption = food_consumption[
    food_consumption["food_category"] == "rice"
]

# Histogram of co2_emission for rice and show plot
rice_consumption["co2_emission"].hist()
plt.show()
print("We can see how to the histogram is right-skewed")


# Use .agg() to calculate the mean and median of co2_emission for rice.
rice_consumption["co2_emission"].agg([np.mean, np.median])

# Given the skew of this data, what measure of central tendency best
# summarizes the kilograms of CO2 emissions per person per year for rice?
# - [ ] Possible answers
# - [ ] Mean
# - [X] Median
# - [ ] Both mean and median
#
# Explanation:
# The mean is substantially higher than the median since it's being pulled up by
# the high values over 100 kg/person/year.
#   mean: 37.59
# median: 15.20
