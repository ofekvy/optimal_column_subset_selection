import time

import pandas as pd

from a_star_search import AStarSearch
from main import get_sparse_matrix

if __name__ == '__main__':
    sparse_matrix = get_sparse_matrix()
    selected_columns_number = 3

    a_star_column_selection = AStarSearch(sparse_matrix)
    selected_columns = a_star_column_selection.run_search(selected_columns_number)

    print(sparse_matrix)
    print("Selected columns:", selected_columns)
    print('approx error: ', a_star_column_selection.approx_error(selected_columns))

    columns_list = [[1, 2, 8], [1, 2, 3]]
    for columns in columns_list:
        print('columns: ', columns)
        print('cost: ', a_star_column_selection.cost_function(columns, selected_columns_number))
        print('approx error: ', a_star_column_selection.approx_error(columns))

    mat_df = pd.read_csv(r"datasets/SPECTF.test")
    matrix = mat_df.to_numpy()
    selected_columns_number = 3

    a_star_column_selection = AStarSearch(matrix)
    start_time = time.time()
    selected_columns = a_star_column_selection.run_search(selected_columns_number)
    end_time = time.time()

    print(matrix)
    print('time: ', end_time - start_time)
    print("Selected columns: ", selected_columns)
    print('Approx error: ', a_star_column_selection.approx_error(selected_columns))
