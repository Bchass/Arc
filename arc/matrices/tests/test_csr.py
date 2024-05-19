import pytest
from arc.matrices import csr_matrix
from arc.matrices import data_types


def test_default_constructor():
    test_input_matrix = csr_matrix(3, 3)
    assert test_input_matrix.rows == 3
    assert test_input_matrix.cols == 3
    expected_matrix = [[0] * 3 for _ in range(3)]

    assert test_input_matrix.matrix == expected_matrix


def test_all():
    test_input_matrix = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[
                                   0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2])
    expected_matrix = [[1, 0, 2], [0, 0, 3], [4, 5, 6]]

    assert test_input_matrix.matrix == expected_matrix


def test_get_cols():
    test_input_matrix = csr_matrix(3, 3).get_cols()

    expected_result = 3

    assert test_input_matrix == expected_result


def test_get_rows():
    test_input_matrix = csr_matrix(3, 3).get_rows()

    expected_result = 3

    assert test_input_matrix == expected_result


def test_get_size():
    test_input_matrix = csr_matrix(3, 3).get_size()

    expected_result = 9

    assert test_input_matrix == expected_result


def test_nnz():
    test_input_matrix = csr_matrix(3, 3, data=[1, 2, 3, 4, 5, 6], row=[
                                   0, 0, 1, 2, 2, 2], col=[0, 2, 2, 0, 1, 2]).nnz()

    expected_result = 6

    assert test_input_matrix == expected_result


def test_multiplication():
    A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[
                   0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

    D = csr_matrix(2, 2, data=[7, 8, 9, 10], row=[
                   0, 0, 1, 1], col=[0, 1, 0, 1])

    assert (A[0, :] == [1, 2] and (A[1, :] == [3, 4] and (A[2, :] == [5, 6])))
    assert (D[0, :] == [7, 8] and (D[1, :] == [9, 10]))

    result_matrix = A.multiply(D)

    assert (result_matrix[0, :] == [25, 28] and (
        result_matrix[1, :] == [57, 64] and (result_matrix[2, :] == [89, 100])))

    with pytest.raises(ValueError):
        A = csr_matrix(3, 3)
        D = csr_matrix(3, 3)

        result_matrix = A.multiply(D)


def test_addition():
    A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[
                   0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

    D = csr_matrix(3, 2, data=[7, 8, 9, 10], row=[
                   0, 0, 1, 1], col=[0, 1, 0, 1])

    assert (A[0, :] == [1, 2] and (A[1, :] == [3, 4] and (A[2, :] == [5, 6])))
    assert (D[0, :] == [7, 8] and (D[1, :] == [9, 10] and (D[2, :] == [0, 0])))

    result_matrix = A.add(D)

    assert (result_matrix[0, :] == [8, 10] and (
        result_matrix[1, :] == [12, 14] and (result_matrix[2, :] == [5, 6])))

    with pytest.raises(ValueError):
        A = csr_matrix(2, 2)
        D = csr_matrix(3, 2)

        result_matrix = A.add(D)


def test_subtraction():
    A = csr_matrix(3, 2, data=[1, 2, 3, 4, 5, 6], row=[
                   0, 0, 1, 1, 2, 2], col=[0, 1, 0, 1, 0, 1])

    D = csr_matrix(3, 2, data=[7, 8, 9, 10], row=[
                   0, 0, 1, 1], col=[0, 1, 0, 1])

    assert (A[0, :] == [1, 2] and (A[1, :] == [3, 4] and (A[2, :] == [5, 6])))
    assert (D[0, :] == [7, 8] and (D[1, :] == [9, 10] and (D[2, :] == [0, 0])))

    result_matrix = A.add(D)

    assert (result_matrix[0, :] == [8, 10] and (
        result_matrix[1, :] == [12, 14] and (result_matrix[2, :] == [5, 6])))

    with pytest.raises(ValueError):
        A = csr_matrix(2, 2)
        D = csr_matrix(3, 2)

        result_matrix = A.add(D)


def test_divison():

    A = csr_matrix(2, 2, data=[4, 8, 12, 16], row=[
                   0, 0, 1, 1], col=[0, 1, 0, 1])

    D = csr_matrix(2, 2, data=[1, 2, 3, 4], row=[0, 0, 1, 1], col=[0, 1, 0, 1])

    assert (A[0, :] == [4, 8] and (A[1, :] == [12, 16]))
    assert (D[0, :] == [1, 2] and (D[1, :] == [3, 4]))

    result_matrix = A.divide(D)

    assert (result_matrix[0, :] == [4, 4] and (
        result_matrix[1, :] == [4, 4]))

    with pytest.raises(ValueError):
        A = csr_matrix(2, 2)
        D = csr_matrix(3, 2)

        result_matrix = A.divide(D)
