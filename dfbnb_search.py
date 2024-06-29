import numpy as np
from sortedcontainers import SortedList
from column_subset_selection import ColumnSubsetSelection


class DFBnB(ColumnSubsetSelection):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.m, self.n = X.shape

    @staticmethod
    def get_orthonormal_basis(S_i):
        Q, _ = np.linalg.qr(S_i)
        non_zero_cols = np.sum(np.abs(S_i), axis=0) > 0
        return Q * non_zero_cols

    def get_b_eigenvalues(self, selected_columns: list) -> np.array:
        X = self.X
        S_i = X[:, selected_columns]
        Q = self.get_orthonormal_basis(S_i)
        X_i = X - Q @ Q.T @ X

        _, eigenvalues, _ = np.linalg.svd(X_i)
        return np.sort(np.real(eigenvalues))

    def cost_function(self, selected_columns: list, k: int) -> float:
        m = self.m
        k_p = len(selected_columns)
        sorted_eigenvalues = self.get_b_eigenvalues(selected_columns)
        f = np.sum(sorted_eigenvalues[:m - k + k_p])
        return f

    def run_search(self, k: int) -> list:
        X, m, n = self.X, self.m, self.n

        stack = [([], float('inf'))] # (selected columns, current cost)
        best_selected_columns = []
        min_cost = float('inf')
        min_pruning_value = float('inf')

        while stack:
            selected_columns, cost = stack.pop()
            k_p = len(selected_columns)

            if k_p == k:
                if cost < min_cost:
                    min_cost = cost
                    best_selected_columns = selected_columns
                continue

            children_list = []
            for i in range(n):
                if i not in selected_columns:
                    curr_selected_columns = selected_columns + [i]
                    curr_cost = self.cost_function(curr_selected_columns, k)
                    pruning_value = (k_p + 2) * curr_cost
                    min_pruning_value = min([min_pruning_value, pruning_value])
                    if curr_cost < min_cost and curr_cost <= min_pruning_value:
                        children_list.append((curr_selected_columns, curr_cost))
            children_list.sort(reverse=True, key=lambda x: x[1])
            stack += children_list

        return best_selected_columns

