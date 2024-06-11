import numpy as np


class ColumnSubsetSelection:
    def __init__(self, X: np.ndarray):
        self.X = X

    @staticmethod
    def get_svd_mats(X: np.ndarray) -> tuple:
        eig, V = np.linalg.eig(X @ X.T)
        eig = np.maximum(eig, 0) # to handle very small negative eigenvalues
        D = np.real(np.diag(eig))
        svd_transformation_mat = D ** 0.5 @ np.real(V.T)
        return D, svd_transformation_mat

    def approx_error(self, selected_columns):
        X = self.X
        S_i = X[:, selected_columns]
        A = np.linalg.pinv(S_i) @ X
        approx = S_i @ A
        return np.linalg.norm(X-approx, 'fro')

    def run_search(self, k: int):
        pass
