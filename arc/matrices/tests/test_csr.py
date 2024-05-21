import pytest
from arc.matrices import csr_matrix
from arc.matrices.data_types import *

def test_default_constructor():
    try:
        matrix_instance = csr_matrix(3, 3)

        assert matrix_instance.rows == 3, f"Expected 3 rows, but got {matrix_instance.rows}"
        assert matrix_instance.cols == 3, f"Expected 3 columns, but got {matrix_instance.cols}"

        expected_matrix = [[0] * 3 for _ in range(3)]
        assert matrix_instance.matrix == expected_matrix, f"Expected {expected_matrix}, but got {matrix_instance.matrix}"

        assert isinstance(matrix_instance, csr_matrix), "The object is not an instance of csr_matrix"

    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        matrix_instance = csr_matrix(3,3)
        with pytest.raises(TypeError):
            value = matrix_instance[0]
    except Exception as e:
        assert False, f"An error orccured: {e}"

def test_call_returns_self():
    try:
        obj = csr_matrix(3,3)
        assert obj() is obj
    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_all():
    try:
        matrix_instance = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2])
        expected_matrix = [[1, 0, 2], [0, 0, 3], [4, 5, 6]]

        assert matrix_instance.matrix == expected_matrix, f"Expected {expected_matrix}, but got {matrix_instance.matrix}"

        assert isinstance(matrix_instance, csr_matrix), "The object is not an instance of csr_matrix"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_length():
    try:
        with pytest.raises(ValueError):

            matrix_instance = csr_matrix(3,3, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 2] , col=[0, 2, 2, 0, 1, 2])
    
    except Exception as e:
        assert False, f"An error occured: {e}"

def test_length_rows_times_cols():
    try:
        with pytest.raises(ValueError):

             matrix_instance = csr_matrix(3,3, data=[1, 2, 3, 4, 5, 6])

    except Exception as e:
        assert False, f"An error occured: {e}"

def test_dtype():
    try:
        matrix_instance = csr_matrix(3,3,dtype=int8)
        assert matrix_instance.dtype is not None, "dtype should not be None"

        expected_str = '[[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int8'
        matrix_str = str(matrix_instance.matrix) + ', dtype=' + matrix_instance.dtype.__name__
        assert matrix_str == expected_str, "String representation should match expected"
    except Exception as e:
        assert False, f"An error occured: {e}"

    try:
        matrix_instance = csr_matrix(3,3)
        assert matrix_instance.dtype is None, "dtype should be None"

        expected_str = '[[0, 0, 0], [0, 0, 0], [0, 0, 0]]'
        matrix_str = str(matrix_instance.matrix)
        assert matrix_str == expected_str, "String representation should match expected"

    except Exception as e:
        assert False, f"An error occured: {e}"

def test_empty_matrix_repr():
    try:
        matrix_instance = csr_matrix(0, 0)

        assert repr(matrix_instance) == '', "Expected '' string representation for an empty matrix"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_get_rows():
    try:
        matrix_instance = csr_matrix(0, 3)
        assert matrix_instance.get_rows() == 0, "Expected 0 rows"

        matrix_instance = csr_matrix(3, 3)
        assert matrix_instance.get_rows() == 3, "Expected 3 rows"

        # Limit test: Performance impact at 100x100
        matrix_instance = csr_matrix(100, 3)
        assert matrix_instance.get_rows() == 100, "Expected 100 rows"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_get_cols():
    try:
        matrix_instance = csr_matrix(3, 0)
        assert matrix_instance.get_cols() == 0, "Expected 0 columns"

        matrix_instance = csr_matrix(3, 3)
        assert matrix_instance.get_cols() == 3, "Expected 3 columns"

        # Limit test: Performance impact at 100x100
        matrix_instance = csr_matrix(3, 100)
        assert matrix_instance.get_cols() == 100, "Expected 100 columns"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_get_size():
    try:
        matrix_instance = csr_matrix(0, 0)
        assert matrix_instance.get_size() == 0, "Expected size 0 for empty matrix"

        matrix_instance = csr_matrix(0, 3)
        assert matrix_instance.get_size() == 0, "Expected size 0 for empty matrix"

        matrix_instance = csr_matrix(3, 0)
        assert matrix_instance.get_size() == 0, "Expected size 0 for empty matrix"

        matrix_instance = csr_matrix(3, 3)
        assert matrix_instance.get_size() == 9, "Expected size 9 for 3x3 matrix"

        # Limit test: Performance impact at 100x100
        matrix_instance = csr_matrix(100, 100)
        assert matrix_instance.get_size() == 10000, "Expected size 10000 for 100x100 matrix"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_nnz():
    try:
        matrix_instance = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2])
        result = matrix_instance.nnz()
        assert result == 6, "Expected 6 non-zero elements"

        matrix_instance = csr_matrix(3, 3)
        result = matrix_instance.nnz()
        assert result == 0, "Expected 0 non-zero elements for an empty matrix"

        matrix_instance = csr_matrix(3, 3, data=[0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = matrix_instance.nnz()
        assert result == 0, "Expected 0 non-zero elements for a matrix with all zeros"

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_toarray():
    try:
        matrix_instance = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2], dtype=int8)
        dense_array, dtype_str = matrix_instance.toarray()
        
        expected_array = [[1, 0, 2], [0, 0, 3], [4, 5, 6]]
        expected_dtype_str = 'int8'

        assert dense_array == expected_array
        assert dtype_str == expected_dtype_str

    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        matrix_instance = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2])
        matrix_instance.toarray()

        expected_array = [[1, 0, 2], [0, 0, 3], [4, 5, 6]]
        assert dense_array == expected_array

    except Exception as e:
        assert False, f"An error occurred: {e}"


def test_multiplication():
    try:
        A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])
        D = csr_matrix(2, 2, data=[7, 8, 9, 10], row=[0, 0, 1, 1], col=[0, 1, 0, 1])

        assert A[0, :] == [1, 2] and A[1, :] == [3, 4] and A[2, :] == [5, 6]
        assert D[0, :] == [7, 8] and D[1, :] == [9, 10]

        result_matrix = A.multiply(D)

        expected_result = csr_matrix(3, 2, data=[25, 28, 57, 64, 89, 100], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert result_matrix.matrix == expected_result.matrix, "Multiplication result is incorrect"
        
    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        A = csr_matrix(3,3)
        D = (3,3)

        with pytest.raises(ValueError):
            A.multiply(D)
        
    except ValueError:
            pass

    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        A = csr_matrix(3, 2, data=[0, 0, 0, 0, 0, 0], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])
        D = csr_matrix(2, 2, data=[0, 0, 0, 0, 0, 0], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert A[0, :] == [0, 0] and A[1, :] == [0, 0] and A[2, :] == [0, 0]
        assert D[0, :] == [0, 0] and D[1, :] == [0, 0]

        with pytest.raises(ValueError):
            result_matrix = A.multiply(D)

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_addition():
    try:
        A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])
        D = csr_matrix(3, 2, data=[7, 8, 9, 10, 0, 0], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert A[0, :] == [1, 2] and A[1, :] == [3, 4] and A[2, :] == [5, 6]
        assert D[0, :] == [7, 8] and D[1, :] == [9, 10] and D[2, :] == [0, 0]

        result_matrix = A.add(D)
        expected_result = csr_matrix(3, 2, data=[8, 10, 12, 14, 5, 6], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert result_matrix.matrix == expected_result.matrix, "The result of addition is incorrect"

        with pytest.raises(ValueError):
            A = csr_matrix(2, 2)
            D = csr_matrix(3, 2)

            result_matrix = A.add(D)

    except Exception as e:
        assert False, f"An error occurred: {e}"
    
    try:
        A = csr_matrix(3,3)
        D = (3,3)

        with pytest.raises(ValueError):
            A.add(D)
        
    except ValueError:
            pass

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_subtraction():
    try:
        A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])
        D = csr_matrix(3, 2, data=[7, 8, 9, 10, 0, 0], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert A[0, :] == [1, 2] and A[1, :] == [3, 4] and A[2, :] == [5, 6]
        assert D[0, :] == [7, 8] and D[1, :] == [9, 10] and D[2, :] == [0, 0]

        result_matrix = A.subtract(D)

        expected_result = csr_matrix(3, 2, data=[-6, -6, -6, -6, 5, 6], row=[0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

        assert result_matrix.matrix == expected_result.matrix, "The result of subtraction is incorrect"

        with pytest.raises(ValueError):
            A = csr_matrix(2, 2)
            D = csr_matrix(3, 2)

            result_matrix = A.subtract(D)

    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        A = csr_matrix(3,3)
        D = (3,3)

        with pytest.raises(ValueError):
            A.subtract(D)
        
    except ValueError:
            pass

    except Exception as e:
        assert False, f"An error occurred: {e}"

def test_division():
    try:
        A = csr_matrix(2, 2, data=[4, 8, 12, 16], row=[0, 0, 1, 1], col=[0, 1, 0, 1])
        D = csr_matrix(2, 2, data=[1, 2, 3, 4], row=[0, 0, 1, 1], col=[0, 1, 0, 1])

        assert A[0, :] == [4, 8] and A[1, :] == [12, 16]
        assert D[0, :] == [1, 2] and D[1, :] == [3, 4]

        result_matrix = A.divide(D)

        expected_result = csr_matrix(2, 2, data=[4, 4, 4, 4], row=[0, 0, 1, 1], col=[0, 1, 0, 1])

        assert result_matrix.matrix == expected_result.matrix, "The result of division is incorrect"

        with pytest.raises(ValueError):
            A = csr_matrix(2, 2)
            D = csr_matrix(3, 2)

            result_matrix = A.divide(D)

        with pytest.raises(ValueError):
            A = csr_matrix(2, 2, data=[4, 8, 12, 16], row=[0, 0, 1, 1], col=[0, 1, 0, 1])
            D = csr_matrix(2, 2, data=[0, 0, 0, 0], row=[0, 0, 1, 1], col=[0, 1, 0, 1])

            result_matrix = A.divide(D)

    except Exception as e:
        assert False, f"An error occurred: {e}"

    try:
        A = csr_matrix(3,3)
        D = (3,3)

        with pytest.raises(ValueError):
            A.divide(D)
        
    except ValueError:
            pass

    except Exception as e:
        assert False, f"An error occurred: {e}"