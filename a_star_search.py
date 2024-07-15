import numpy as np
from sortedcontainers import SortedList

from column_subset_selection import ColumnSubsetSelection


class AStarSearch(ColumnSubsetSelection):

    def run_search(self, selected_columns_number: int):
        initial_cost = np.trace(self.matrix @ self.matrix.T)
        start_state = (initial_cost, [], (None, self.diagonal_root_matrix))
        open_set = SortedList([start_state], key=lambda x: -x[0])
        closed_set = []
        min_pruning_value = float('inf')

        while open_set:
            current_node = open_set.pop(-1)
            current_cost, selected_columns, current_matrices = current_node

            if len(selected_columns) == selected_columns_number:
                return selected_columns, self.generated_vertices

            closed_set.append(selected_columns)
            for col in range(self.number_columns):
                if col not in selected_columns:
                    new_selected_columns = selected_columns + [col]
                    new_cost, new_matrices = self.efficient_cost_function(selected_columns, col,
                                                                          selected_columns_number,
                                                                          current_matrices)
                    state = (new_cost, new_selected_columns, new_matrices)
                    pruning_value = (len(selected_columns) + 2) * new_cost
                    min_pruning_value = min([min_pruning_value, pruning_value])

                    if new_selected_columns not in closed_set and new_cost <= min_pruning_value:
                        open_set.add(state)

        return [], self.generated_vertices
