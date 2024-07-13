import time

import numpy as np
import pandas as pd

from dfbnb_without_ordering import DFBnB
from main import get_sparse_matrix

if __name__ == '__main__':
    # sparse_matrix = get_sparse_matrix()
    # selected_columns_number = 3
    #
    # dfbnb_column_selection = DFBnB(sparse_matrix)
    # selected_columns = dfbnb_column_selection.run_search(selected_columns_number)
    #
    # print(sparse_matrix)
    # print("Selected columns:", selected_columns)
    #
    # columns_list = [selected_columns, [5, 6, 8], [1, 2, 8], [1, 2, 3]]
    # for columns in columns_list:
    #     print('columns: ', columns)
    #     print('approx error: ', dfbnb_column_selection.approx_error(columns))

    # mat_df = pd.read_csv(r"datasets/SPECTF.test")
    # matrix = mat_df.to_numpy()
    # selected_columns_number = 3
    matrix = np.random.rand(30, 20)
    selected_columns_number = 5

    dfbnb_column_selection = DFBnB(matrix)
    start_time = time.time()
    selected_columns = dfbnb_column_selection.run_search(selected_columns_number)
    end_time = time.time()

    print(matrix)
    print('time: ', end_time - start_time)
    print("Selected columns: ", selected_columns)
    print('Approx error: ', dfbnb_column_selection.approx_error(selected_columns))
