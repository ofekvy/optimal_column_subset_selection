from abc import abstractmethod, ABC
import time
import numpy as np


def get_orthonormal_basis(selected_matrix):
    orthonormal_matrix, _ = np.linalg.qr(selected_matrix)
    non_zero_cols = np.sum(np.abs(selected_matrix), axis=0) > 0

    return orthonormal_matrix * non_zero_cols


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
        self.diagonal_root_matrix, self.transformation_matrix = self.get_diagonal_and_transformation_mats()
        self.generated_vertices = 0  # Counter for generated vertices
        self.cost_function_time = 0
        self.storing_and_calculating_next_node_time = 0

    def get_diagonal_and_transformation_mats(self) -> tuple:
        eigen_values, eigen_vectors = np.linalg.eigh(self.matrix @ self.matrix.T)
        eig_idx = eigen_values > 1e-8
        eigen_values = eigen_values[eig_idx]
        diagonal_root_matrix = np.real(np.diag(eigen_values))
        transformation_matrix = diagonal_root_matrix ** 0.5 @ np.real(eigen_vectors[:, eig_idx].T)
        return diagonal_root_matrix, transformation_matrix

    def cost_function(self, selected_columns: list, selected_columns_number: int) -> float:
        sorted_eigenvalues = self.get_eigenvalues(selected_columns)
        adjusted_eigenvalues_sum = np.sum(
            sorted_eigenvalues[:len(sorted_eigenvalues) - selected_columns_number + len(selected_columns)])
        self.generated_vertices += 1
        return float(adjusted_eigenvalues_sum)

    def efficient_cost_function(self, previously_selected_columns: list, new_selected_column: int,
                                selected_columns_number: int, parent_matrices: list) -> tuple:
        start_time = time.time()
        self.generated_vertices += 1

        selected_columns = previously_selected_columns + [new_selected_column]
        current_selected_columns_number = len(selected_columns)
        parent_orthogonal_matrix, parent_special_matrix = parent_matrices
        selected_matrix = self.matrix[:, selected_columns]

        orthonormal_mat, orthonormal_vec = self.get_updated_orthonormal_matrix(
            parent_orthogonal_matrix,
            selected_matrix
        )
        transformation_vec = self.transformation_matrix @ orthonormal_vec
        special_matrix = parent_special_matrix - transformation_vec @ transformation_vec.T
        cost = self.calculate_cost_from_special_matrix(special_matrix, selected_columns_number,
                                                       current_selected_columns_number)
        new_matrices = [orthonormal_mat, special_matrix]
        end_time = time.time()
        self.cost_function_time += end_time - start_time
        return cost, new_matrices

    @staticmethod
    def calculate_cost_from_special_matrix(special_matrix, selected_columns_number,
                                           current_selected_columns_number):
        eigenvalues, _ = np.linalg.eigh(special_matrix)
        eigenvalues = np.abs(np.real(eigenvalues))
        sorted_eigenvalues = np.sort(eigenvalues)
        cost = np.sum(sorted_eigenvalues[:len(sorted_eigenvalues) - selected_columns_number
                                          + current_selected_columns_number])
        return cost

    @staticmethod
    def get_updated_orthonormal_matrix(parent_orthogonal_matrix, selected_matrix):
        if parent_orthogonal_matrix is not None:
            projection = selected_matrix - parent_orthogonal_matrix @ parent_orthogonal_matrix.T @ selected_matrix
        else:
            projection = selected_matrix
        orthonormal_vec = projection / np.linalg.norm(projection) if np.linalg.norm(projection) > 0 else projection
        if parent_orthogonal_matrix is not None:
            orthonormal_mat = np.concatenate([parent_orthogonal_matrix, orthonormal_vec], axis=1)
        else:
            orthonormal_mat = orthonormal_vec
        return orthonormal_mat, orthonormal_vec

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

        return np.sort(np.abs(np.real(eigenvalues)) ** 2)

    @abstractmethod
    def run_search(self, selected_columns_number: int) -> list:
        pass
