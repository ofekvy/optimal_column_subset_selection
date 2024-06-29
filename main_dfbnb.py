import numpy as np
import random
from dfbnb_search import DFBnB
import time
import pandas as pd


if __name__=='__main__':
    random.seed(42)
    sparse_matrix = np.zeros((10, 10))
    sparse_matrix[0, [1, 3, 5]] = [10, 20, 30]
    sparse_matrix[3, [2, 4, 6]] = [15, 25, 35]
    sparse_matrix[6, [3, 8, 9]] = [5, 10, 15]
    sparse_matrix[7, 8] = 20
    k = 3  # Number of columns to select
    dfbnb_column_selection = DFBnB(sparse_matrix)
    selected_columns = dfbnb_column_selection.run_search(k)
    print(sparse_matrix)
    print("Selected columns:", selected_columns)
    columns_list = [selected_columns, [5, 6, 8], [1, 2, 8], [1, 2, 3]]
    for columns in columns_list:
        print('columns: ', columns)
        # print('elgenvalues: ', a_star_column_selection.get_b_eigenvalues([i]))
        print('approx error: ', dfbnb_column_selection.approx_error(columns))

    mat_df = pd.read_csv('random_matrix.csv')
    X = mat_df.to_numpy()
    k = 15  # Number of columns to select
    dfbnb_column_selection = DFBnB(X)
    start_time = time.time()
    selected_columns = dfbnb_column_selection.run_search(k)
    end_time = time.time()
    print(X)
    print('time: ', end_time - start_time)
    print("Selected columns: ", selected_columns)
    print('Approx error: ', dfbnb_column_selection.approx_error(selected_columns))