import pandas as pd
import numpy as np

def calculate_psi(expected, actual, bins=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    
    Parameters:
    expected (array-like): Reference distribution (training data)
    actual (array-like): Current distribution (new data)
    bins (int): Number of bins to use for the distributions
    
    Returns:
    psi_value (float): The PSI value
    """
    # Create bins using quantiles
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_bins = pd.cut(expected, np.percentile(expected, breakpoints), include_lowest=True)
    actual_bins = pd.cut(actual, np.percentile(expected, breakpoints), include_lowest=True)

    # Calculate the percentage of observations in each bin
    expected_percents = expected_bins.value_counts()
    actual_percents = actual_bins.value_counts()
    
    # Ensure no zero values in the arrays to avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate the PSI value
    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    
    return psi_value


# Let's create some example data to calculate PSI
reference_data = np.array([100, 200, 150, 50, 300, 250, 400, 300, 500, 700])
current_data = np.array([80, 220, 140, 60, 310, 260, 420, 310, 510, 720])

psi_value = calculate_psi(reference_data, current_data)

print(f"PSI value: {psi_value}")


