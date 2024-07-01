from column_subset_selection import ColumnSubsetSelection


class DFBnB(ColumnSubsetSelection):

    def run_search(self, selected_columns_number: int) -> list:
        stack = [([], float('inf'))]  # (selected columns, current cost)
        best_selected_columns = []
        min_cost = float('inf')
        min_pruning_value = float('inf')

        while stack:
            selected_columns, cost = stack.pop()

            if len(selected_columns) == selected_columns_number:
                if cost < min_cost:
                    min_cost = cost
                    best_selected_columns = selected_columns
                continue

            children_list = []
            for i in range(self.number_columns):
                if i not in selected_columns:
                    curr_selected_columns = selected_columns + [i]
                    curr_cost = self.cost_function(curr_selected_columns, selected_columns_number)
                    pruning_value = (len(selected_columns) + 2) * curr_cost
                    min_pruning_value = min([min_pruning_value, pruning_value])

                    if curr_cost < min_cost and curr_cost <= min_pruning_value:
                        children_list.append((curr_selected_columns, curr_cost))
            children_list.sort(reverse=True, key=lambda x: x[1])
            stack += children_list

        return best_selected_columns
