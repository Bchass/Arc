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
    row = [0, 0, 1, 2, 2, 2]
    col = [0, 2, 2, 0, 1, 2]
    data = [1, 2, 3, 4, 5, 6]
    test_input_matrix = csr_matrix(3, 3, data=data, row=row, col=col)
    expected_matrix = [[1, 0, 2], [0, 0, 3], [4, 5, 6]]

    assert test_input_matrix.matrix == expected_matrix


def test_data():
    data = [1, 2, 3, 4, 5, 6]
    test_input_matrix = csr_matrix(3, 3, data=data)
    expected_matrix = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]

    assert test_input_matrix.matrix == expected_matrix


def test_nnz():
    row = [0, 0, 1, 2, 2, 2]
    col = [0, 2, 2, 0, 1, 2]
    data = [1, 2, 3, 4, 5, 6]
    test_input_matrix = csr_matrix(3, 3, data=data, row=row, col=col).nnz()

    expected_result = 6

    assert test_input_matrix == expected_result
