
import numpy as np
import pandas as pd

from machine_learning_vault.ECDF.ecdf import calculate_ecdf
from scipy.stats import ks_2samp

# Step 1: Prepare your data
# Suppose you have a binary classification model, and you have predicted 
# probabilities for the positive class and the actual labels for your test set.

# The probabilities that the model predicted
predicted_probabilities = [0.9, 0.8, 0.6, 0.4, 0.7, 0.2, 0.3, 0.1, 0.5, 0.55]
# The real target value 
actual_labels = [1, 1, 0, 0, 1, 0, 0, 1, 0, 1]

predicted_probabilities = np.array(predicted_probabilities)
actual_labels = np.array(actual_labels)


# Step 2: Separate Predictions by Class
# Separate the predicted probabilities into two groups based on the actual labels.

positive_probabilities = predicted_probabilities[actual_labels == 1]
negative_probabilities = predicted_probabilities[actual_labels == 0]

# Step 3: Calculate the ECDF for feach group
# The Empirical Cumulative Distribution Function (ECDF) is a step function.
positive_x, positive_ecdf = calculate_ecdf(positive_probabilities)

negative_x, negative_ecdf = calculate_ecdf(negative_probabilities)

# Calculate the ks using a third party library
ks_statistic, p_value = ks_2samp(positive_probabilities, negative_probabilities)

print(f"""
KS Calculated with scipy
-----------------------
KS statistic represent the maximum difference between the tho EDFs. A hihger
KS indicates a better separation between the two classes.
    ks_statistic: {ks_statistic}    

The P-value helps determine the significance of the KS statistic. A low 
p-value (typically < 0.5) suggests that the difference between the two
distributions is statistically significant
    p_value: {p_value}
""")


# Step 3: Calculate the KS Statistic manually
def ks_statistic(ecdf1_x, ecdf1_y, ecdf2_x, ecdf2_y):
    # Combine all x values
    all_x = np.sort(np.concatenate((ecdf1_x, ecdf2_x)))
    # Interpolate ECDF values at combined x values
    ecdf1_y_interp = np.interp(all_x, ecdf1_x, ecdf1_y, left=0, right=1)
    ecdf2_y_interp = np.interp(all_x, ecdf2_x, ecdf2_y, left=0, right=1)
    # Calculate the maximum difference
    max_diff = np.max(np.abs(ecdf1_y_interp - ecdf2_y_interp))
    return max_diff

ks_stat_manually_calculated = ks_statistic(
            positive_x, positive_ecdf, 
            negative_x, negative_ecdf 
        )

print(f"""
KS Statistic calculated manually
--------------------------------
{ks_stat_manually_calculated}
""")
