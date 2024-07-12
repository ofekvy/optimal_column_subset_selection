import random
import time

import numpy as np
import pandas as pd

from a_star_search import AStarSearch
from dfbnb_search import DFBnB


def get_sparse_matrix():
    """
        Generate a 10*10 matrix with zeros expect some specific values.
    """
    random.seed(42)
    sparse_matrix = np.zeros((10, 10))
    sparse_matrix[0, [1, 3, 5]] = [10, 20, 30]
    sparse_matrix[3, [2, 4, 6]] = [15, 25, 35]
    sparse_matrix[6, [3, 8, 9]] = [5, 10, 15]
    sparse_matrix[7, 8] = 20

    return sparse_matrix


def test_a_star(matrix, selected_columns_number):
    a_star_column_selection = AStarSearch(matrix)
    start_time = time.time()
    selected_columns = a_star_column_selection.run_search(selected_columns_number)
    end_time = time.time()

    print('selected_columns A* : ', selected_columns)
    print('time A* : ', end_time - start_time)
    print('Approx error A*: ', a_star_column_selection.approx_error(selected_columns))


def test_dfbnb(matrix, selected_columns_number):
    dfbnb = DFBnB(matrix)
    start_time = time.time()
    selected_columns = dfbnb.run_search(selected_columns_number)
    end_time = time.time()

    print('selected_columns DFBnB : ', selected_columns)
    print('time DFBnB : ', end_time - start_time)
    print('Approx error DFBnB: ', dfbnb.approx_error(selected_columns))


if __name__ == '__main__':
    # Case Where A* is better than DFBNB
    print('\nRandom Matrix')
    matrix_data_frame = pd.read_csv('datasets/random_matrix.csv')
    matrix = matrix_data_frame.to_numpy()
    selected_columns_number = 5
    print(f'Matrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')

    test_a_star(matrix, selected_columns_number)
    test_dfbnb(matrix, selected_columns_number)

    # Case Where A* is not better than DFBNB
    print('\nUniform Matrix')
    matrix = np.ones((30, 20))
    selected_columns_number = 7
    print(f'Matrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')

    test_a_star(matrix, selected_columns_number)
    test_dfbnb(matrix, selected_columns_number)

    print('\nReal Dataset')
    matrix_data_frame = pd.read_csv('datasets/SPECTF.test')
    matrix = matrix_data_frame.to_numpy()
    selected_columns_number = 5
    print(f'Matrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')

    test_a_star(matrix, selected_columns_number)
    test_dfbnb(matrix, selected_columns_number)
