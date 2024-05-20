import pytest
from arc.matrices import dia_matrix
from arc.matrices.data_types import *

# TODO: Need to figure out a better example with dtype
# TODO: Refactor test cases

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

def test_set_element():
    try:
        test_input_matrix = dia_matrix(3, 4)
        test_input_matrix.set_element(0, 0, 3)
        expected_matrix = [[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert test_input_matrix.matrix == expected_matrix, "Matrix should match expected after setting element"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo:
        test_set_element = dia_matrix(3, 4)
        test_set_element.set_element(3, 0, 0)
    assert "Index out of range" in str(excinfo.value), "Exception message should indicate index out of range"

    with pytest.raises(ValueError) as excinfo:
        test_set_element = dia_matrix(3, 4)
        test_set_element.set_element(1, 5, 0)
    assert "Index out of range" in str(excinfo.value), "Exception message should indicate index out of range"

    with pytest.raises(ValueError) as excinfo:
        test_set_element = dia_matrix(3, 4)
        test_set_element.set_element(1, 2, 0)
    assert "Can only set elements on diagonal" in str(excinfo.value), "Exception message should indicate only diagonal elements can be set"

def test_slice():
    try:
        test_input_matrix = dia_matrix(3, 4)
        test_input_matrix.set_element(0, 0, 3)

        assert test_input_matrix[0, :] == [3, 0, 0, 0], "Slice should match expected"
        assert test_input_matrix[0, 0] == int(3), "Single element access should match expected"
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
        A = dia_matrix(2, 2)
        A.set_element(1, 0, 2)
        A.set_element(0, 0, 3)
        A.set_element(1, 1, 1)
        A.set_element(0, 1, 1)

        D = dia_matrix(2, 2)
        D.set_element(0, 0, 2)
        D.set_element(1, 1, 4)

        assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1]), "Matrix A slices should match expected"
        assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4]), "Matrix D slices should match expected"

        result_matrix = A.multiply(D)

        assert (result_matrix[0, :] == [6, 4]) and (result_matrix[1, :] == [4, 4]), "Result matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)
        _ = A.multiply(D)
    assert "Matrices must be of the same size for multiplication" in str(excinfo1.value), "Exception message should indicate incompatible sizes for multiplication"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(3, 4)
        D = dia_matrix(3, 4)
        _ = A.multiply(D)
    assert "Number of columns in the first matrix does not equal to the number of rows in the second matrix" in str(excinfo2.value), "Exception message should indicate different shapes for multiplication"

def test_addition():
    try:
        A = dia_matrix(2, 2)
        A.set_element(1, 0, 2)
        A.set_element(0, 0, 3)
        A.set_element(1, 1, 1)
        A.set_element(0, 1, 1)

        D = dia_matrix(2, 2)
        D.set_element(0, 0, 2)
        D.set_element(1, 1, 4)

        assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1]), "Matrix A slices should match expected"
        assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4]), "Matrix D slices should match expected"

        result_matrix = A.add(D)

        assert (result_matrix[0, :] == [5, 1]) and (result_matrix[1, :] == [2, 5]), "Result matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)
        _ = A.add(D)
    assert "Matrices must be of the same size for addition" in str(excinfo1.value), "Exception message should indicate incompatible sizes for addition"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(3, 4)
        D = dia_matrix(3, 4)
        _ = A.add(D)
    assert "Number of columns in the first matrix does not equal to the number of rows in the second matrix" in str(excinfo2.value), "Exception message should indicate different shapes for addition"

def test_subtraction():
    try:
        A = dia_matrix(3, 3)
        A.set_element(0, 0, 1)
        A.set_element(1, 1, 4)
        A.set_element(2, 2, 3)

        D = dia_matrix(3, 3)
        D.set_element(0, 0, 1)
        D.set_element(1, 1, 3)
        D.set_element(2, 2, 2)

        assert (A[0, :] == [1, 0, 0]) and (A[1, :] == [0, 4, 0]) and (A[2, :] == [0, 0, 3]), "Matrix A slices should match expected"
        assert (D[0, :] == [1, 0, 0]) and (D[1, :] == [0, 3, 0]) and (D[2, :] == [0, 0, 2]), "Matrix D slices should match expected"

        result_matrix = A.subtract(D)

        assert (result_matrix[0, :] == [0, 0, 0]) and (result_matrix[1, :] == [0, 1, 0]) and (result_matrix[2, :] == [0, 0, 1]), "Result matrix should match expected"
    except Exception as e:
        pytest.fail(f"Basic test failed with exception: {e}")

    with pytest.raises(ValueError) as excinfo1:
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)
        _ = A.subtract(D)
    assert "Matrices must be of the same size for subtraction" in str(excinfo1.value), "Exception message should indicate incompatible sizes for subtraction"

    with pytest.raises(ValueError) as excinfo2:
        A = dia_matrix(3, 3)
        A.set_element(0, 0, -1)
        D = dia_matrix(3, 3)
        _ = A.subtract(D)
    assert "Positive definite matrix cannot have negative diagonal elements" in str(excinfo2.value), "Exception message should indicate negative diagonal element"    

# TODO: Update this after format is fixed
def test_toarray():

    A = dia_matrix(3, 4).toarray()
    expected_matrix = """
    [[0, 0, 0, 0],
    [0, 0, 0, 0], 
    [0, 0, 0, 0]]"""
    actual_str = ''.join(str(A).split())
    expected_str = ''.join(expected_matrix.split())

    assert actual_str == expected_str
