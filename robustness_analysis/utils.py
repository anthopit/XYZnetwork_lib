def compare_arrays(arr1, arr2, message="This value is not supported"):
    # Check if every element in arr2 is also in arr1
    for element in arr2:
        if element not in arr1:
            raise ValueError(f"Value {element}: {message}")