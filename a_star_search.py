import numpy as np
from heapq import heappush, heappop
from sortedcontainers import SortedList


class ColumnSubsetAStarSearch:
    def __init__(self, X: np.ndarray):
        self.X = X
        self.n, self.m = X.shape
        self.D, self.svd_stransformation_mat = self.get_svd_stransformation_mat(X)

    @staticmethod
    def get_svd_mats(X: np.ndarray) -> tuple:
        V, D, _ = np.linalg.svd(X)
        svd_transformation_mat = D ** 0.5 @ V.T
        return D, svd_transformation_mat

    @staticmethod
    def get_q_y(Q_p: np.ndarray, y: np.ndarray)  -> np.ndarray:
        if Q_p is not None:
            q_y = Q_p @ Q_p.T @ y
        else:
            q_y = y
        return q_y / np.linalg.norm(q_y)

    @staticmethod
    def get_z_y(svd_transformation_mat: np.ndarray, q_y: np.ndarray) -> np.ndarray:
        return svd_transformation_mat @ q_y

    @staticmethod
    def get_H_y(H_p: np.ndarray, z_y: np.ndarray) -> np.ndarray:
        return H_p - z_y * z_y.T

    def cost_function(self, parents_columns: list, selected_column: int, Q_p: np.ndarray, H_p: np.ndarray) -> list:
        X = self.X
        svd_transformation_mat = self.svd_stransformation_mat
        y = X[:, selected_column]
        q_y = self.get_q_y(Q_p, y)
        z_y = self.get_z_y(svd_transformation_mat, q_y)
        H_y = self.get_H_y(H_p, z_y)

        # if not selected_columns:
        #     return np.linalg.norm(A, 'fro')
        # A_subset = A[:, selected_columns]
        # U, s, Vt = np.linalg.svd(A_subset, full_matrices=False)
        # approx = (U * s) @ Vt
        # return np.linalg.norm(A - approx, 'fro')

    @staticmethod
    def heuristic_function(A: np.ndarray, selected_columns: list, k: int) -> list:
        # Estimate the remaining cost to reach k columns
        remaining_columns = k - len(selected_columns)
        return remaining_columns * np.mean(np.linalg.norm(A, axis=0))

    def optimal_column_subset_selection(self, k: int) -> list:
        A, m, n = self.X, self.m, self.n
        start_state = (0, [], None)  # (cost, selected_columns, Q_i)
        open_set = SortedList([start_state])
        closed_set = SortedList()

        while open_set:
            current_node = open_set.pop(0)
            current_cost, selected_columns = current_node

            if len(selected_columns) == k:
                return selected_columns

            closed_set.append(selected_columns)
            for col in range(m):
                if col not in selected_columns:
                    new_selected_columns = selected_columns + [col]
                    new_cost = self.cost_function(A, new_selected_columns)
                    estimated_total_cost = new_cost - self.heuristic_function(A, new_selected_columns, k)
                    state = (estimated_total_cost, new_selected_columns)

                    if tuple(new_selected_columns) not in closed_set:
                        open_set.append(state)

        return []