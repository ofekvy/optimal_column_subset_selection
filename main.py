import numpy as np
import random
from a_star_search import AStarSearch
from dfbnb_search import DFBnB
import time
import pandas as pd


def test_a_star(X, k):
    a_star_column_selection = AStarSearch(X)
    start_time = time.time()
    selected_columns = a_star_column_selection.run_search(k)
    end_time = time.time()
    print('selected_columns A* : ',  selected_columns)
    print('time A* : ', end_time - start_time)
    print('Approx error A*: ', a_star_column_selection.approx_error(selected_columns))


def test_dfbnb(X, k):
    dfbnb = DFBnB(X)
    start_time = time.time()
    selected_columns = dfbnb.run_search(k)
    end_time = time.time()
    print('selected_columns DFBnB : ',  selected_columns)
    print('time DFBnB : ', end_time - start_time)
    print('Approx error DFBnB: ', dfbnb.approx_error(selected_columns))
    print('')


if __name__=='__main__':
    # Case Where A* is better than DFBNB
    print('Random Matrix')
    mat = pd.read_csv('datasets/random_matrix.csv')
    X = mat.to_numpy()
    k = 5
    print(f'Matrix shape: {X.shape}, k = {k}')
    test_a_star(X, k)
    test_dfbnb(X, k)

    # Case Where A* is not better than DFBNB
    print('Uniform Matrix')
    X = np.ones((30, 20))
    k = 7
    print(f'Matrix shape: {X.shape}, k = {k}')
    test_a_star(X, k)
    test_dfbnb(X, k)

    #Real dataset
    print('Uniform Matrix')
    mat = pd.read_csv('datasets/SPECTF.test')
    X = mat.to_numpy()
    k = 5
    print(f'Matrix shape: {X.shape}, k = {k}')
    test_a_star(X, k)
    test_dfbnb(X, k)

