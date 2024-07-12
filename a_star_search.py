import numpy as np
from sortedcontainers import SortedList

from column_subset_selection import ColumnSubsetSelection


class AStarSearch(ColumnSubsetSelection):

    def run_search(self, selected_columns_number: int) -> list:
        initial_cost = np.trace(self.matrix @ self.matrix.T)
        start_state = (initial_cost, [])
        open_set = SortedList([start_state], key=lambda x: -x[0])
        closed_set = []
        min_pruning_value = float('inf')

        while open_set:
            current_node = open_set.pop(-1)
            current_cost, selected_columns = current_node

            if len(selected_columns) == selected_columns_number:
                return selected_columns

            closed_set.append(selected_columns)
            for col in range(self.number_columns):
                if col not in selected_columns:
                    new_selected_columns = selected_columns + [col]
                    new_cost = self.cost_function(new_selected_columns, selected_columns_number)
                    state = (new_cost, new_selected_columns)
                    pruning_value = (len(selected_columns) + 2) * new_cost
                    min_pruning_value = min([min_pruning_value, pruning_value])

                    if new_selected_columns not in closed_set and new_cost <= min_pruning_value:
                        open_set.add(state)

        return []
