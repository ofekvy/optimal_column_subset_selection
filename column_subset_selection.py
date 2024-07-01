from abc import abstractmethod, ABC

import numpy as np


def get_orthonormal_basis(selected_matrix):
    orthonormal_matrix, _ = np.linalg.qr(selected_matrix)
    non_zero_cols = np.sum(np.abs(selected_matrix), axis=0) > 0

    return orthonormal_matrix[:, non_zero_cols]


def get_svd_mats(matrix: np.ndarray) -> tuple:
    eigenvalues, eigenvectors = np.linalg.eig(matrix @ matrix.T)
    eigenvalues = np.maximum(eigenvalues, 0)  # to handle very small negative eigenvalues
    D = np.real(np.diag(eigenvalues))
    svd_transformation_mat = D ** 0.5 @ np.real(eigenvectors.T)

    return D, svd_transformation_mat


class ColumnSubsetSelection(ABC):

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.number_rows, self.number_columns = matrix.shape

    def cost_function(self, selected_columns: list, selected_columns_number: int) -> float:
        sorted_eigenvalues = self.get_eigenvalues(selected_columns)
        adjusted_eigenvalues_sum = np.sum(
            sorted_eigenvalues[:self.number_rows - selected_columns_number + len(selected_columns)])

        return float(adjusted_eigenvalues_sum)

    def approx_error(self, selected_columns):
        selected_basis = self.matrix[:, selected_columns]
        pseudo_inverse_basis = np.linalg.pinv(selected_basis) @ self.matrix
        approximation_matrix = selected_basis @ pseudo_inverse_basis

        return np.linalg.norm(self.matrix - approximation_matrix, 'fro')

    def get_eigenvalues(self, selected_columns: list) -> np.array:
        selected_matrix = self.matrix[:, selected_columns]
        orthonormal_basis = get_orthonormal_basis(selected_matrix)
        residual_matrix = self.matrix - orthonormal_basis @ orthonormal_basis.T @ self.matrix
        _, eigenvalues, _ = np.linalg.svd(residual_matrix)

        return np.sort(np.real(eigenvalues))

    @abstractmethod
    def run_search(self, selected_columns_number: int) -> list:
        pass
