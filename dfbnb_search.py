from column_subset_selection import ColumnSubsetSelection


class DFBnB(ColumnSubsetSelection):

    @staticmethod
    def update_selected_columns(selected_columns: list,
                                parent_matrices: list,
                                selected_columns_number: int,
                                number_columns: int):
        if len(selected_columns) < selected_columns_number:
            last_column = selected_columns[-1] if selected_columns else 0
            for i in range(last_column, number_columns):
                if i not in selected_columns:
                    selected_columns.append(i)
                    return
        else:
            while len(selected_columns):
                selected_columns[-1] += 1
                if selected_columns[-1] < number_columns:
                    return
                else:
                    parent_matrices.pop()
                    selected_columns.pop()

    @staticmethod
    def prune_path(selected_columns: list, parent_matrices: list,
                   number_columns: int):
        while len(selected_columns):
            selected_columns[-1] += 1
            if selected_columns[-1] < number_columns:
                return  # Valid child node found, exit prune_path
            else:
                parent_matrices.pop()
                selected_columns.pop()

    def find_first_solution(self, selected_columns_number: int) -> tuple:
        best_selected_columns = []
        solution_min_cost = float('inf')
        for j in range(selected_columns_number):

            min_cost = float('inf')
            min_column = -1
            for i in range(self.number_columns):
                if i not in best_selected_columns:
                    curr_selected_columns = best_selected_columns + [i]
                    curr_cost = self.cost_function(curr_selected_columns, selected_columns_number)
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        min_column = i
            best_selected_columns.append(min_column)
            solution_min_cost = min_cost
        self.generated_vertices += len(best_selected_columns)

        return best_selected_columns, solution_min_cost

    def run_search(self, selected_columns_number: int):
        selected_columns = [0]
        parent_matrices = [[None, self.diagonal_root_matrix]]
        best_selected_columns, min_cost = self.find_first_solution(selected_columns_number)
        min_pruning_value = float('inf')

        while len(selected_columns):
            cost, curr_matrices = self.efficient_cost_function(selected_columns[:-1], selected_columns[-1],
                                                               selected_columns_number, parent_matrices[-1])
            parent_matrices.append(curr_matrices)
            if len(selected_columns) == selected_columns_number:
                if cost < min_cost:
                    min_cost = cost
                    best_selected_columns = selected_columns.copy()
                self.update_selected_columns(selected_columns, parent_matrices, selected_columns_number,
                                             self.number_columns)
            else:
                if cost > min_cost or cost > min_pruning_value:
                    self.prune_path(selected_columns, parent_matrices, self.number_columns)
                else:
                    self.update_selected_columns(selected_columns, parent_matrices, selected_columns_number,
                                                 self.number_columns)
            min_pruning_value = min([min_pruning_value, cost * (len(selected_columns) + 1)])

        return best_selected_columns, self.generated_vertices
