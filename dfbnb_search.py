from column_subset_selection import ColumnSubsetSelection
import time


class DFBnB(ColumnSubsetSelection):

    @staticmethod
    def update_selected_columns(selected_columns: list,
                                parent_matrices: list,
                                selected_columns_number: int,
                                number_columns: int):
        if len(selected_columns) < selected_columns_number:
            for i in range(selected_columns[-1]+1,number_columns):
                selected_columns.append(i)
                return
            parent_matrices.pop()
            while len(selected_columns)-1:
                parent_matrices.pop()
                selected_columns.pop()
                selected_columns[-1] += 1
                if selected_columns[-1] < number_columns:
                    return
            parent_matrices.pop()
            selected_columns.pop()
        else:
            while len(selected_columns):
                selected_columns[-1] += 1
                second_last_element = selected_columns[-2] if len(selected_columns) > 1 else -1
                if selected_columns[-1] > second_last_element:
                    parent_matrices.pop()
                    if selected_columns[-1] < number_columns:
                        return
                    else:
                        selected_columns.pop()

    @staticmethod
    def prune_path(selected_columns: list, parent_matrices: list,
                   number_columns: int):
        parent_matrices.pop()
        while len(selected_columns):
            selected_columns[-1] += 1
            second_last_element = selected_columns[-2] if len(selected_columns) > 1 else -1
            if selected_columns[-1] > second_last_element:
                if selected_columns[-1] < number_columns:
                    return
                selected_columns.pop()
                if len(parent_matrices) >= len(selected_columns):
                    parent_matrices.pop()

    def find_first_solution(self, selected_columns_number: int) -> tuple:
        best_selected_columns = []
        solution_min_cost = float('inf')
        parent_matrices = [None, self.diagonal_root_matrix]
        for j in range(selected_columns_number):

            min_cost = float('inf')
            min_column = -1
            min_parent_matrices = []
            for i in range(self.number_columns):
                if i not in best_selected_columns:
                    curr_cost, curr_matrices = self.efficient_cost_function(best_selected_columns, i,
                                                                            selected_columns_number, parent_matrices)

                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        min_column = i
                        min_parent_matrices = curr_matrices
            best_selected_columns.append(min_column)
            solution_min_cost = min_cost
            parent_matrices = min_parent_matrices
        return best_selected_columns, solution_min_cost

    def run_search(self, selected_columns_number: int, with_first_search=True):
        selected_columns = [0]
        parent_matrices = [[None, self.diagonal_root_matrix]]
        if with_first_search:
            best_selected_columns, min_cost = self.find_first_solution(selected_columns_number)
        else:
            best_selected_columns, min_cost = [], float('inf')

        while len(selected_columns):
            cost, curr_matrices = self.efficient_cost_function(selected_columns[:-1], selected_columns[-1],
                                                               selected_columns_number, parent_matrices[-1])
            start_time_calculating_next_node = time.time()
            parent_matrices.append(curr_matrices)
            if len(selected_columns) == selected_columns_number:
                if cost < min_cost:
                    min_cost = cost
                    best_selected_columns = selected_columns.copy()
                self.update_selected_columns(selected_columns, parent_matrices, selected_columns_number,
                                             self.number_columns)
            else:
                if cost > min_cost:
                    self.prune_path(selected_columns, parent_matrices, self.number_columns)
                else:
                    self.update_selected_columns(selected_columns, parent_matrices, selected_columns_number,
                                                 self.number_columns)
            end_time_calculating_next_node = time.time()
            self.storing_and_calculating_next_node_time += end_time_calculating_next_node - start_time_calculating_next_node

        return best_selected_columns, self.generated_vertices
