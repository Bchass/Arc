import pytest
from arc.arc.dia import dia_matrix
from arc.arc.data_types import *

# TODO: Need to figure out a better example with dtype
# TODO: Refactor test cases


def test_default_constrcutor():

    test_input_matrix = dia_matrix(3, 4, dtype=int8)
    assert test_input_matrix.size == 3
    assert test_input_matrix.shape == 4
    assert test_input_matrix.dtype == int8
    expected_matrix = [[0] * 4 for _ in range(3)]

    assert test_input_matrix.matrix == expected_matrix


def test_set_element():

    test_input_matrix = dia_matrix(3, 4)
    test_input_matrix.set_element(0, 0, 3)
    assert test_input_matrix.matrix == [
        [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    expected_matrix = [[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    assert test_input_matrix.matrix == expected_matrix

    with pytest.raises(ValueError):
        test_set_element = dia_matrix(3, 4)
        test_set_element.set_element(1, 0, 0)
    with pytest.raises(ValueError):
        test_set_element.set_element(3, 3, 0)


def test_slice():

    test_input_matrix = dia_matrix(3, 4)
    test_input_matrix.set_element(0, 0, 3)

    assert test_input_matrix[0, :] == [3, 0, 0, 0]
    assert test_input_matrix[0, 0] == int(3)

    with pytest.raises(IndexError):
        test_slice = dia_matrix(3, 4)
        test_slice[3, 3]


def test_shape():

    A = dia_matrix(2, 2)
    assert A.get_shape() == 2


def test_size():

    A = dia_matrix(2, 2)
    assert A.get_size() == 2


def test_multiplication():

    A = dia_matrix(2, 2)
    A.set_element(1, 0, 2)
    A.set_element(0, 0, 3)
    A.set_element(1, 1, 1)
    A.set_element(0, 1, 1)

    D = dia_matrix(2, 2)
    D.set_element(0, 0, 2)
    D.set_element(1, 1, 4)

    assert (A[0, :] == [3, 1]) and (A[1, :] == [2, 1])
    assert (D[0, :] == [2, 0]) and (D[1, :] == [0, 4])

    result_matrix = A.multiply(D)

    assert (result_matrix[0, :] == [6, 4]) and (result_matrix[1, :] == [4, 4])

    with pytest.raises(ValueError):
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)

        result_matrix = A.multiply(D)

        assert result_matrix

    with pytest.raises(ValueError):
        A = dia_matrix(3, 4)
        D = dia_matrix(3, 4)

        result_matrix = A.multiply(D)


def test_addition():

    A = dia_matrix(2, 2)
    A.set_element(1, 0, 2)
    A.set_element(0, 0, 3)
    A.set_element(1, 1, 1)
    A.set_element(0, 1, 1)

    D = dia_matrix(2, 2)
    D.set_element(0, 0, 2)
    D.set_element(1, 1, 4)

    assert (A[0, :] == [3, 1] and (A[1, :] == [2, 1]))
    assert (D[0, :] == [2, 0] and (D[1, :] == [0, 4]))

    result_matrix = A.add(D)

    assert (result_matrix[0, :] == [5, 1] and (result_matrix[1, :] == [2, 5]))

    with pytest.raises(ValueError):
        A = dia_matrix(2, 3)
        D = dia_matrix(3, 4)

        result_matrix = A.add(D)

        assert result_matrix

    with pytest.raises(ValueError):
        A = dia_matrix(3, 4)
        D = dia_matrix(3, 4)

        result_matrix = A.add(D)


def test_subtraction():

    A = dia_matrix(3, 3)
    A.set_element(0, 0, 1)
    A.set_element(1, 1, 4)
    A.set_element(2, 2, 3)

    D = dia_matrix(3, 3)
    D.set_element(0, 0, 1)
    D.set_element(1, 1, 3)
    D.set_element(2, 2, 2)

    assert (A[0, :] == [1, 0, 0] and (
        A[1, :] == [0, 4, 0] and (A[2, :] == [0, 0, 3])))
    assert (D[0, :] == [1, 0, 0] and (
        D[1, :] == [0, 3, 0] and (D[2, :] == [0, 0, 2])))

    result_matrix = A.subtract(D)

    assert (result_matrix[0, :] == [0, 0, 0] and (
        result_matrix[1, :] == [0, 1, 0] and (result_matrix[2, :] == [0, 0, 1])))
    

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
