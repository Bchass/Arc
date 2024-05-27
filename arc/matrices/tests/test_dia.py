import pytest, sys, io
from arc.matrices import dia_matrix
from arc.matrices.data_types import *
import numpy as np


def test_default_constructor():
    try:
        test_input_matrix = dia_matrix(3, 4, dtype=int8)
        assert test_input_matrix.size == 3, "Size should be 3"
        assert test_input_matrix.shape == 4, "Shape should be 4"
        assert test_input_matrix.dtype == int8, "Data type should be int8"
        expected_matrix = [[0] * 4 for _ in range(3)]
        assert test_input_matrix.matrix == expected_matrix, "Matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(3, 4)
        assert test_input_matrix.dtype is None, "Data type should be None if not specified"
    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(3, 2)
        assert test_input_matrix.shape == 2, "Shape should be 2"
        expected_matrix = [[0] * 2 for _ in range(3)]
        assert test_input_matrix.matrix == expected_matrix, "Matrix should match expected"
    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(3, shape=4)
        assert test_input_matrix.shape == 4, "Shape should be 4"
        assert test_input_matrix.dtype is None, "Data type should be None"
        expected_matrix = [[0] * 4 for _ in range(3)]
        assert test_input_matrix.matrix == expected_matrix, "Matrix should match expected"
    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(2,2, data=[1,2])
        expected_matrix = [[1,0], [0,2]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"

    except Exception as e:
       pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(2,2, data=[1,2,3,4])
        expected_matrix = [[1,3], [4,2]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"

    except Exception as e:
       pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(2,2, data=[0,0,3,4])
        expected_matrix = [[0,3], [4,0]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"

    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(2,2, data=[3,4,0,0])
        expected_matrix = [[3,0], [0,4]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"

    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(4,4, data=[1,2,3,4])
        expected_matrix = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"
    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")

    try:
        test_input_matrix = dia_matrix(4, 4, data=[1, 2, 3, 4], offsets=[0,2,-1,1,2])
        expected_matrix = [[1, 3, 0, 0], [4, 1, 3, 0], [2, 4, 1, 3], [0, 2, 4, 1]]

        assert test_input_matrix.matrix == expected_matrix, f"Expected {expected_matrix}, but got {test_input_matrix.matrix}"
    except Exception as e:
        pytest.fail(f"Edge case test failed with exception: {e}")
    
    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output

        test_input_matrix = dia_matrix(5, 5, data=[1, 2, 3, 4], offsets=[6])
        printed_output = captured_output.getvalue().strip()

        sys.stdout = sys.__stdout__

        assert printed_output == "Ignoring offset 6, it's out of bounds"
    except Exception as e:
        pytest.fail(f"Edge case test failed with unexpected exception: {e}")


def test_getitem():
    try:
        test_input_matrix = dia_matrix(3, 3,data=[1,2,3])

        assert test_input_matrix[0,
                                 0] == 1, "Value at index (0, 0) should match"
        assert test_input_matrix[1,
                                 1] == 2, "Value at index (1, 1) should match"
        assert test_input_matrix[2,
                                 2] == 3, "Value at index (2, 2) should match"

        # Test slice indexing
        assert test_input_matrix[0, :] == [
            1, 0, 0], "Slice at row 0 should match"
        assert test_input_matrix[1, :] == [
            0, 2, 0], "Slice at row 1 should match"
        assert test_input_matrix[2, :] == [
            0, 0, 3], "Slice at row 2 should match"

        # Test out of bounds index
        with pytest.raises(IndexError) as excinfo1:
            _ = test_input_matrix[3, 0]
        assert "list index out of range" in str(
            excinfo1.value), "Exception should be raised for out of bounds index"

        # Test unsupported index type
        with pytest.raises(IndexError) as excinfo2:
            _ = test_input_matrix[0]
        assert "Unsupported index type" in str(
            excinfo2.value), "Exception should be raised for unsupported index type"

    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")


def test_call_returns_self():
    try:
        obj = dia_matrix(3, 3)
        assert obj() is obj
    except Exception as e:
        assert False, f"An error occurred: {e}"


def test_empty_matrix_repr():
    try:
        matrix_instance = dia_matrix(0, 0)

        assert repr(
            matrix_instance) == '', "Expected '' string representation for an empty matrix"

    except Exception as e:
        assert False, f"An error occurred: {e}"


def test_dtype():
    try:
        matrix_instance = dia_matrix(3, 3, dtype=int8)
        assert matrix_instance.dtype is not None, "dtype should not be None"

        expected_str = '[[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int8'
        matrix_str = str(matrix_instance.matrix) + \
            ', dtype=' + matrix_instance.dtype.__name__
        assert matrix_str == expected_str, "String representation should match expected"
    except Exception as e:
        assert False, f"An error occured: {e}"

    try:
        matrix_instance = dia_matrix(3, 3)
        assert matrix_instance.dtype is None, "dtype should be None"

        expected_str = '[[0, 0, 0], [0, 0, 0], [0, 0, 0]]'
        matrix_str = str(matrix_instance.matrix)
        assert matrix_str == expected_str, "String representation should match expected"

    except Exception as e:
        assert False, f"An error occured: {e}"
        
def test_slice():
    try:
        test_input_matrix = dia_matrix(3, 4, data=[3])

        assert test_input_matrix[0, :] == [
            3, 0, 0, 0], "Slice should match expected"
        assert test_input_matrix[0, 0] == int(
            3), "Single element access should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")


def test_shape():
    try:
        A = dia_matrix(2, 2)
        assert A.get_shape() == 2, "Shape should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    try:
        B = dia_matrix(3, 5)
        assert B.get_shape() == 5, "Shape should match specified value"
    except Exception as e:
        pytest.fail(f"Test case failed with exception: {e}")


def test_size():
    try:
        A = dia_matrix(2, 2)
        assert A.get_size() == 2, "Size should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    try:
        B = dia_matrix(3, 5)
        assert B.get_size() == 3, "Size should match specified value"
    except Exception as e:
        pytest.fail(f"Test case failed with exception: {e}")

    try:
        C = dia_matrix(4)
        assert C.get_size() == 4, "Size should equal specified value"
    except Exception as e:
        pytest.fail(f"Test case failed with exception: {e}")


def test_multiplication():
    try:
        A = dia_matrix(2, 2, data=[3,1,1,2])

        D = dia_matrix(2, 2, data=[2,4])

        assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1]
                                        ), "Matrix A slices should match expected"
        assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4]
                                        ), "Matrix D slices should match expected"

        result_matrix = A.multiply(D)

        assert (result_matrix[0, :] == [6, 4]) and (
            result_matrix[1, :] == [4, 4]), "Result matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 2)
        D = (2, 2)
        A.multiply(D)
    assert "Instance is not a dia_matrix" in str(
        excinfo1.value), "Exception message should indicate Instance is not a dia_matrix"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(2, 2)
        D = dia_matrix(3, 4)
        _ = A.multiply(D)
    assert "Matrices must be of the same size for multiplication" in str(
        excinfo2.value), "Exception message should indicate incompatible sizes for multiplication"

    with pytest.raises(ValueError) as excinfo3:
        A = dia_matrix(3, 3)
        D = dia_matrix(3, 4)
        _ = A.multiply(D)
    assert "Number of columns in the first matrix does not equal to the number of rows in the second matrix" in str(
        excinfo3.value), "Exception message should indicate Number of columns in the first matrix does not equal to the number of rows in the second matrix"


def test_addition():
    try:
        A = dia_matrix(2, 2, data=[3,1,1,2])

        D = dia_matrix(2, 2, data=[2,4])

        assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1]
                                        ), "Matrix A slices should match expected"
        assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4]
                                        ), "Matrix D slices should match expected"

        result_matrix = A.add(D)

        assert (result_matrix[0, :] == [5, 1]) and (
            result_matrix[1, :] == [2, 5]), "Result matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 2)
        D = (2, 2)
        A.add(D)
    assert "Instance is not a dia_matrix" in str(
        excinfo1.value), "Exception message should indicate Instance is not a dia_matrix"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(2, 4)
        D = dia_matrix(3, 4)
        _ = A.add(D)
    assert "Matrices must be of the same size for addition" in str(
        excinfo2.value), "Exception message should indicate incompatible sizes for addition"

    with pytest.raises(ValueError) as excinfo3:
        A = dia_matrix(3, 3)
        D = dia_matrix(3, 4)
        _ = A.add(D)
    assert "Number of columns in the first matrix does not equal to the number of rows in the second matrix" in str(
        excinfo3.value), "Exception message should indicate Number of columns in the first matrix does not equal to the number of rows in the second matrix"


def test_subtraction():
    try:
        A = dia_matrix(3,3,data=[1,2,3])

        D = dia_matrix(3, 3,data=[1,3,2])

        assert (A[0, :] == [1, 0, 0]) and (A[1, :] == [0, 2, 0]) and (
            A[2, :] == [0, 0, 3]), "Matrix A slices should match expected"
        assert (D[0, :] == [1, 0, 0]) and (D[1, :] == [0, 3, 0]) and (
            D[2, :] == [0, 0, 2]), "Matrix D slices should match expected"

        result_matrix = A.subtract(D)

        assert (result_matrix[0, :] == [0, 0, 0]) and (result_matrix[1, :] == [0, -1, 0]) and (
            result_matrix[2, :] == [0, 0, 1]), "Result matrix should match expected"

    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 2)
        D = (2, 2)
        A.subtract(D)
    assert "Instance is not a dia_matrix" in str(
        excinfo1.value), "Exception message should indicate Instance is not a dia_matrix"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)
        _ = A.subtract(D)
    assert "Matrices must be of the same size for subtraction" in str(
        excinfo2.value), "Exception message should indicate incompatible sizes for subtraction"

    with pytest.raises(ValueError) as excinfo3:
        A = dia_matrix(3, 3,data=[-1,0,0])

        D = dia_matrix(3, 3,data=[1,0,0])
        _ = A.subtract(D)
    assert "Positive definite matrix cannot have negative diagonal elements" in str(
        excinfo3.value), "Exception message should indicate negative diagonal elements in matrix A"

    with pytest.raises(ValueError) as excinfo4:
        A = dia_matrix(3, 3,data=[1,0,0])

        D = dia_matrix(3, 3,data=[-1,0,0])
        _ = A.subtract(D)
    assert "Positive definite matrix cannot have negative diagonal elements" in str(
        excinfo4.value), "Exception message should indicate negative diagonal elements in matrix D"


def test_division():
    # Test division by zero encountered
    try:
        A = dia_matrix(3, 3,data=[1,4,3])

        D = dia_matrix(3, 3,data=[1,0,2])

        with pytest.raises(ValueError):
            A.divide(D)
    except Exception as e:
        pytest.fail(f"Test for division by zero failed with exception: {e}")
    # Test matrices of different sizes
    try:
        A = dia_matrix(3, 3,data=[1,4,3])

        D = dia_matrix(2, 2,data=[1,3])

        with pytest.raises(ValueError):
            A.divide(D)
    except Exception as e:
        pytest.fail(
            f"Test for matrices of different sizes failed with exception: {e}")
    # Test matrices not being dia_matrix instances
    try:
        A = dia_matrix(2, 2)

        D = (2, 2)

        with pytest.raises(ValueError):
            A.divide(D)
    except Exception as e:
        pytest.fail(
            f"Test for matrices not being dia_matrix instances failed with exception: {e}")
    # Test matrices not compatible for division
    try:
        A = dia_matrix(3, 3, data=[1,4,3])

        D = dia_matrix(3, 3, data=[1,3,2])

        with pytest.raises(ValueError):
            A.divide(D)
    except Exception as e:
        pytest.fail(
            f"Test for matrices not compatible for division failed with exception: {e}")

def test_toarray():
    try:
        matrix_instance = dia_matrix(3, 3).toarray()

        expected_array = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]], dtype='int64')

        assert np.array_equal(matrix_instance, expected_array)

    except Exception as e:
        assert False, f"An error occurred: {e}"

