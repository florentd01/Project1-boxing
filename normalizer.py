#!/usr/bin/env python3

import pickle
import os
import sys


import numpy as np


DATA_DIR = sys.argv[1]


def read_pickle(file_name):
    """
    Reads the contents of a pickle file into a variable.
    """
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def save_normalized(variable, file_name):
    """
    Saves a variable to a new pickle file with '_normalized' appended to the original file name.
    """
    file_name_without_extension, file_extension = os.path.splitext(file_name)
    normalized_file_name = f'{file_name_without_extension}_normalized{file_extension}'

    with open(normalized_file_name, 'wb') as file:
        pickle.dump(variable, file)


def split_matrix_by_three(matrix):
    """
    Split matrices to three by axis
    """
    return [[row[i::3] for row in matrix] for i in range(3)]


def concat_matrices(matrix1, matrix2, matrix3):
    """
    Reverse of the `split_matrix_by_three`
    """
    return [sum(zip(row1, row2, row3), ()) for row1, row2, row3 in zip(matrix1, matrix2, matrix3)]


def normalize_matrix(matrix):
    """
    Normalizes a 2D matrix using min-max normalization.
    """
    # Convert to numpy array for ease of computation
    matrix = np.array(matrix)

    # Find min and max values in the matrix
    min_value = matrix.min()
    max_value = matrix.max()

    # Avoid division by zero in case all values are the same
    if max_value - min_value == 0:
        return matrix

    # Apply min-max normalization
    normalized_matrix = (matrix - min_value) / (max_value - min_value)

    return normalized_matrix


def normalize_by_three(matrix):
    """
    Splits matrix to x,y,z
    Normalizes by axis
    Concats
    Returns result
    """
    x, y, z = split_matrix_by_three(matrix)

    x = normalize_matrix(x)
    y = normalize_matrix(y)
    z = normalize_matrix(z)

    matrix_normalized = concat_matrices(x, y, z)
    return matrix_normalized


def normalize_pickle(pickle_name):
    """
    Reads, normalize and saves pickle file
    """
    pickle_matrix = read_pickle(pickle_name)
    new_pickle_matrix = []

    for two_d_array in pickle_matrix:
        two_d_array_normalized = normalize_by_three(two_d_array)
        new_pickle_matrix.append(two_d_array_normalized)
    save_normalized(new_pickle_matrix, pickle_name)



if __name__ == "__main__":
    for pickle_name in os.listdir(DATA_DIR):
        if pickle_name.endswith('_landmarks.pkl'):
            normalize_pickle(os.path.join(DATA_DIR, pickle_name))