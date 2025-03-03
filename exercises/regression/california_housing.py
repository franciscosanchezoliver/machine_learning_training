"""
This exercise is based on the following video:
https://www.youtube.com/watch?v=TrzUlo4BImM

This video explains the different metrics for regression problems.


"""

from sklearn.datasets import fetch_california_housing

import pandas as pd


data = fetch_california_housing()
X = data["data"]
y = data["target"]
print("hello")


pd.DataFrame(data["data"], columns=data["feature_names"])
