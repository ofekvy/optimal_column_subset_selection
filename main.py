import random
import time
import tracemalloc

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

    tracemalloc.start()
    start_time = time.time()

    selected_columns, number_of_vertices = a_star_column_selection.run_search(selected_columns_number)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    memory_usage = peak / (1024 ** 2)  # Convert to MB
    tracemalloc.stop()

    print('selected_columns A* : ', selected_columns)
    print('Time A* : ', end_time - start_time)
    print('Calculating cost function Time: ', a_star_column_selection.cost_function_time)
    print('Storing and calculating next node time: ', a_star_column_selection.storing_and_calculating_next_node_time)
    print('Memory usage A* (MB) :', memory_usage)
    print('Number of vertices A* :', number_of_vertices)
    print('Approx error A*: ', a_star_column_selection.approx_error(selected_columns))


def test_dfbnb(matrix, selected_columns_number):
    dfbnb = DFBnB(matrix)

    tracemalloc.start()
    start_time = time.time()
    selected_columns, number_of_vertices = dfbnb.run_search(selected_columns_number)
    end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    memory_usage = peak / (1024 ** 2)  # Convert to MB
    tracemalloc.stop()

    print('selected_columns DFBnB : ', selected_columns)
    print('Time DFBnB : ', end_time - start_time)
    print('Calculating cost function time: ', dfbnb.cost_function_time)
    print('Calculating next node time: ', dfbnb.storing_and_calculating_next_node_time)
    print('Memory usage DFBnB (MB) :', memory_usage)
    print('Number of vertices DFBnB :', number_of_vertices)
    print('Approx error DFBnB: ', dfbnb.approx_error(selected_columns))


def compare_random_matrix():
    print('\nRandom Matrix')
    matrix_data_frame = pd.read_csv('datasets/random_matrix.csv')
    matrix = matrix_data_frame.to_numpy()

    for selected_columns_number in range(1, 7):
        print(f'\nMatrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')
        test_a_star(matrix, selected_columns_number)
        test_dfbnb(matrix, selected_columns_number)


def compare_SPECTF_matrix():
    print('\nSPECTF Dataset')
    matrix_data_frame = pd.read_csv('datasets/SPECTF.test')
    matrix = matrix_data_frame.to_numpy()
    for selected_columns_number in range(1, 5):
        print(f'Matrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')
        test_a_star(matrix, selected_columns_number)
        test_dfbnb(matrix, selected_columns_number)


def compare_libras_movement_matrix():
    print('\nMovement Libras Dataset')
    matrix_data_frame = pd.read_csv('datasets/movement_libras.data')
    matrix = matrix_data_frame.to_numpy()
    for selected_columns_number in range(1, 5):
        print(f'Matrix shape: {matrix.shape}, Selected columns number = {selected_columns_number}')
        test_a_star(matrix, selected_columns_number)
        test_dfbnb(matrix, selected_columns_number)


if __name__ == '__main__':
    compare_random_matrix()
    print("")
    compare_SPECTF_matrix()
    print("")
    compare_libras_movement_matrix()
