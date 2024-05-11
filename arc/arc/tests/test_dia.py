import pytest
from arc.arc.dia import dig_matrix
from arc.arc.data_types import *

# TODO: Need to figure out a better example with dtype
# TODO: Refactor test cases


def test_default_constrcutor():

    test_input_matrix = dig_matrix(3, 4, dtype=int8)
    assert test_input_matrix.size == 3
    assert test_input_matrix.shape == 4
    assert test_input_matrix.dtype == int8
    expected_matrix = [[0] * 4 for _ in range(3)]

    assert test_input_matrix.matrix == expected_matrix


def test_set_element():

    test_input_matrix = dig_matrix(3, 4)
    test_input_matrix.set_element(0, 0, 3)
    assert test_input_matrix.matrix == [
        [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    expected_matrix = [[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    assert test_input_matrix.matrix == expected_matrix

    with pytest.raises(ValueError):
        test_set_element = dig_matrix(3, 4)
        test_set_element.set_element(1, 0, 0)
    with pytest.raises(ValueError):
        test_set_element.set_element(3, 3, 0)


def test_slice():

    test_input_matrix = dig_matrix(3, 4)
    test_input_matrix.set_element(0, 0, 3)

    assert test_input_matrix[0, :] == [3, 0, 0, 0]
    assert test_input_matrix[0, 0] == int(3)

    with pytest.raises(IndexError):
        test_slice = dig_matrix(3, 4)
        test_slice[3, 3]


def test_multiplication():

    A = dig_matrix(2, 2)
    A.set_element(1, 0, 2)
    A.set_element(0, 0, 3)
    A.set_element(1, 1, 1)
    A.set_element(0, 1, 1)

    D = dig_matrix(2, 2)
    D.set_element(0, 0, 2)
    D.set_element(1, 1, 4)

    assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1])
    assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4])

    result_matrix = A.multiply(D)

    assert (result_matrix[0, :] == [6, 4]) and (result_matrix[1, :] == [4, 4])

    with pytest.raises(ValueError):
        A = dig_matrix(2, 3)
        D = dig_matrix(3, 4)

        result_matrix = A.multiply(D)

        assert result_matrix


def test_shape():

    A = dig_matrix(2, 2)
    assert A.get_shape() == 2


def test_size():

    A = dig_matrix(2, 2)
    assert A.get_size() == 2


# TODO: Update this after format is fixed
def test_toarray():

    A = dig_matrix(3, 4).toarray()
    expected_matrix = """
    [[0, 0, 0, 0],
    [0, 0, 0, 0], 
    [0, 0, 0, 0]]"""
    actual_str = ''.join(str(A).split())
    expected_str = ''.join(expected_matrix.split())

    assert actual_str == expected_str
