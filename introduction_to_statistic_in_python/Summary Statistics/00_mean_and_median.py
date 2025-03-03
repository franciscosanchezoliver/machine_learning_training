"""
Mean and median
---------------
In this chapter, you'll be working with the food_consumption dataset from 
2018 Food Carbon Footprint Index by nu3. The food_consumption dataset 
contains the number of kilograms of food consumed per person per year in 
each country and food category (consumption), and its carbon 
footprint (co2_emissions) measured in kilograms of carbon dioxide, or CO2.

In this exercise, you'll compute measures of center to compare food 
consumption in the US and Belgium using your pandas and numpy skills.

Instructions:
- Calculate the mean of food consumption in the usa_consumption DataFrame.
- Calculate the median of food consumption in the usa_consumption DataFrame.
"""

import pandas as pd
import numpy as np

df_food = pd.read_csv("data/food_consumption.csv")


# Subset country for USA: usa_consumption
usa_consumption = df_food[df_food["country"] == "USA"]

# Calculate mean consumption in USA
print(np.mean(usa_consumption["consumption"]))

# Calculate median consumption in USA
print(np.median(usa_consumption["consumption"]))
