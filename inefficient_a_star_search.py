import numpy as np
from sortedcontainers import SortedList


class InefficientColumnSubsetAStarSearch:
    def __init__(self, X: np.ndarray):
        self.X = X
        self.m, self.n = X.shape

    @staticmethod
    def get_svd_mats(X: np.ndarray) -> tuple:
        eig, V = np.linalg.eig(X @ X.T)
        eig = np.maximum(eig, 0) # to handle very small negative eigenvalues
        D = np.real(np.diag(eig))
        svd_transformation_mat = D ** 0.5 @ np.real(V.T)
        return D, svd_transformation_mat

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

        eigenvalues, _ = np.linalg.eig(X_i @ X_i.T)
        return np.sort(np.real(eigenvalues))

    def cost_function(self, selected_columns: list, k: int) -> float:
        m = self.m
        k_p = len(selected_columns)
        sorted_eigenvalues = self.get_b_eigenvalues(selected_columns)
        f = np.sum(sorted_eigenvalues[:m-k+k_p])
        return f

    def approx_error(self, selected_columns):
        X = self.X
        S_i = X[:, selected_columns]
        A = np.linalg.pinv(S_i) @ X
        approx = S_i @ A
        return np.linalg.norm(X-approx, 'fro')


    def run_search(self, k: int) -> list:
        X, m, n = self.X, self.m, self.n
        initial_cost = np.trace(X @ X.T)
        start_state = (initial_cost, [])
        open_set = SortedList([start_state])
        closed_set = []

        while open_set:
            current_node = open_set.pop(0)
            current_cost, selected_columns = current_node

            if len(selected_columns) == k:
                return selected_columns

            closed_set.append(selected_columns)
            for col in range(n):
                if col not in selected_columns:
                    new_selected_columns = selected_columns + [col]
                    new_cost = self.cost_function(new_selected_columns, k)
                    state = (new_cost, new_selected_columns)

                    if new_selected_columns not in closed_set:
                        open_set.add(state)

        return []