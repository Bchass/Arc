import pytest
from arc.linear_alg.BLAS import arc_BLAS


def test_sdot():
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    expected_result = 32.0

    result = arc_BLAS.sdot(x, y)
    assert result == expected_result
