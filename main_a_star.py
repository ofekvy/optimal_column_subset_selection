import numpy as np
import random
from inefficient_a_star_search import InefficientColumnSubsetAStarSearch
from a_star_search import ColumnSubsetAStarSearch
import time


if __name__=='__main__':
    random.seed(42)

    X = np.array([np.linspace(1, 1 + i + 2*random.random(), 5) for i in range(10)])
    sparse_matrix = np.zeros((10, 10))
    sparse_matrix[0, [1, 3, 5]] = [10, 20, 30]
    sparse_matrix[3, [2, 4, 6]] = [15, 25, 35]
    sparse_matrix[6, [3, 8, 9]] = [5, 10, 15]
    sparse_matrix[7, 8] = 20
    k = 3  # Number of columns to select
    a_star_column_selection = ColumnSubsetAStarSearch(sparse_matrix)
    selected_columns = a_star_column_selection.run_search(k)
    print(sparse_matrix)
    print("Selected columns:", selected_columns)

    # X = np.array([np.linspace(1, 1 + i + 2 * random.random(), 5) for i in range(10)])
    X = np.random.rand(30, 20)
    k = 5  # Number of columns to select
    a_star_column_selection = ColumnSubsetAStarSearch(X)
    start_time = time.time()
    selected_columns = a_star_column_selection.run_search(k)
    end_time = time.time()
    print(X)
    print('time: ',end_time - start_time)
    print("Selected columns:", selected_columns)




