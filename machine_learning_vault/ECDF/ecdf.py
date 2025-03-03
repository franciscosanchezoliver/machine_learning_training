
import numpy as np
import matplotlib.pyplot as plt



def calculate_ecdf(data: list[int]):

    data = np.array(data)

    # 1. Sort Data
    sorted_data = np.sort(data)

    ecdf_values = []

    distinct_values = np.unique(sorted_data)

    for each_value in distinct_values:
        values_less_or_equal = sorted_data[sorted_data <= each_value]
        cumulative_proportion = len(values_less_or_equal) / len(data)
        cumulative_proportion = round(cumulative_proportion, 2)
        ecdf_values.append(cumulative_proportion)
        message = f"Values less or equal than [{each_value}] ({cumulative_proportion}) => {values_less_or_equal}"
        print(message)
        
    return distinct_values, ecdf_values 

def print_ecdf(values, ecdf_values):
    # Plot the ECDF in a graph

    plt.step(values, ecdf_values)
    plt.xlabel("Data Values")
    plt.ylabel("ECDF")
    plt.title("Empirical Cumulative Distribution Function (ECDF)")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

    values, ecdf =  calculate_ecdf(data)
    print_ecdf(values, ecdf)