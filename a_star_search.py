import numpy as np
from sortedcontainers import SortedList
from column_subset_selection import ColumnSubsetSelection


class ColumnSubsetAStarSearch(ColumnSubsetSelection):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.m, self.n = X.shape
        self.D, self.svd_stransformation_mat = self.get_svd_mats(X)

    @staticmethod
    def get_q_y(Q_p: np.ndarray, y: np.ndarray) -> tuple:
        if Q_p is not None:
            q_y_t = y - Q_p @ Q_p.T @ y
        else:
            q_y_t = y
        q_y = q_y_t / np.linalg.norm(q_y_t) if np.linalg.norm(q_y_t) > 0 else q_y_t
        if Q_p is not None:
            Q_y = np.concatenate([Q_p, q_y], axis=1)
        else:
            Q_y = q_y
        return q_y, Q_y

    @staticmethod
    def get_z_y(svd_transformation_mat: np.ndarray, q_y: np.ndarray):
        return svd_transformation_mat @ q_y

    @staticmethod
    def get_H_y(H_p: np.ndarray, z_y: np.ndarray) -> np.ndarray:
        return H_p - z_y @ z_y.T

    @staticmethod
    def get_f_y(H_y, k, k_p, m):
        _, eigenvalues, _ = np.linalg.svd(H_y)
        eigenvalues = np.real(eigenvalues)
        sorted_eigenvalues = np.sort(eigenvalues)
        f = np.sum(sorted_eigenvalues[:m - k + k_p + 1])
        return f

    def cost_function(self, parents_columns: list, selected_column: int, Q_p: np.ndarray, H_p: np.ndarray, k: int) -> tuple:
        X, m, D, svd_transformation_mat = self.X, self.m, self.D, self.svd_stransformation_mat
        y = np.expand_dims(X[:, selected_column], 1)
        k_p = len(parents_columns)

        q_y, Q_y = self.get_q_y(Q_p, y)
        z_y = self.get_z_y(svd_transformation_mat, q_y)
        H_y = self.get_H_y(H_p, z_y)
        f_y = self.get_f_y(H_y, k, k_p, m)
        return f_y, Q_y, H_y

    def run_search(self, k: int) -> list:
        X, D, m, n = self.X, self.D, self.m, self.n
        initial_cost = np.trace(D)
        start_state = (initial_cost, [], None, D)  # (cost, selected_columns, Q_c, H_c)
        open_set = SortedList([start_state])
        closed_set = SortedList()

        while open_set:
            current_node = open_set.pop(0)
            current_cost, selected_columns, Q_c, H_c = current_node
            num_of_columns = len(selected_columns)

            if len(selected_columns) == k:
                return selected_columns

            P = (num_of_columns + 1) * current_cost
            closed_set.add(selected_columns)
            for col in range(n):
                if col not in selected_columns:
                    new_selected_columns = selected_columns + [col]
                    new_cost, new_Q, new_H = self.cost_function(selected_columns, col, Q_c, H_c, k)
                    state = (new_cost, new_selected_columns, new_Q, new_H)

                    if new_selected_columns not in closed_set and new_cost <= P:
                        open_set.add(state)

        return []