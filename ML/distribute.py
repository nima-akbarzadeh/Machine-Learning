from itertools import product
import numpy as np


def distribute_budget(T, N):
    # Create an iterable with values ranging from 1 to T (inclusive)
    iterable = range(T + 1)

    # Generate all possible combinations of length N with repetitions
    combinations = list(product(iterable, repeat=N))

    # # Generate all possible combinations of distributing T into N parts
    # combinations = list(itertools.combinations_with_replacement(range(T + 1), N))

    # Filter combinations where the sum is equal to T
    valid_combinations = [combo for combo in combinations if sum(combo) <= T]

    return np.array(valid_combinations)


def print_result_matrix(result_matrix):
    for row in result_matrix:
        print(row)


if __name__ == "__main__":
    # Example usage:
    T = int(input("Enter the budget (T): "))
    N = int(input("Enter the number of machines (N): "))

    matrix = distribute_budget(T, N)
    vector = np.linspace(1 / N, 1, N)
    multiplied_result = np.sum(matrix*vector, axis=1)
    unique_values = np.unique(multiplied_result)

    red_indices = []
    for j in range(len(unique_values)-1):
        if np.abs(unique_values[j] - unique_values[j+1]) < 0.5/N:
            red_indices.append(j+1)

    # Create a boolean mask with True at indices to remove
    mask = np.ones(len(unique_values), dtype=bool)
    mask[red_indices] = False

    # Create a new array without the elements at specified indices
    new_array = unique_values[mask]

    print(new_array)
