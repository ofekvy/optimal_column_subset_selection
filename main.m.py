


if __name__=='__main__':
    A = np.random.rand(10, 5)  # Random 10x5 matrix
    k = 3  # Number of columns to select
    selected_columns = optimal_column_subset_selection(A, k)
    print("Selected columns:", selected_columns)