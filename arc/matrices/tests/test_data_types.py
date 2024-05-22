from arc.matrices.data_types import *
import pytest


def test_int8_valid_value():
    try:
        value = 42
        int_obj = int8(value)
        assert int_obj.value == value
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int8_out_of_bounds():
    try:
        with pytest.raises(ValueError):
            int8(128)

        with pytest.raises(ValueError):
            int8(-129)
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int8_repr():
    try:
        value = 5
        int_obj = int8(value)
        assert repr(int_obj) == f'int8({value})'
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int16_valid_value():
    try:
        value = 3000
        int_obj = int16(value)
        assert int_obj.value == value
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int16_out_of_bounds():
    try:
        with pytest.raises(ValueError):
            int16(32768)

        with pytest.raises(ValueError):
            int16(-32769)
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int16_repr():
    try:
        value = 20000
        int_obj = int16(value)
        assert repr(int_obj) == f'int16({value})'
    except Exception as e:
        assert False, f"An error occurred: {e}"


def test_int32_valid_value():
    try:
        value = 3245678
        int_obj = int32(value)
        assert int_obj.value == value
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int32_out_of_bounds():
    try:
        with pytest.raises(ValueError):
            int32(2147483648)

        with pytest.raises(ValueError):
            int32(-2147483649)
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int32_repr():
    try:
        value = 2147483645
        int_obj = int32(value)
        assert repr(int_obj) == f'int32({value})'
    except Exception as e:
        assert False, f"An error occurred: {e}"


def test_int64_valid_value():
    try:
        value = 92233720368547807
        int_obj = int64(value)
        assert int_obj.value == value
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int64_out_of_bounds():
    try:
        with pytest.raises(ValueError):
            int32(9223372036854775808)

        with pytest.raises(ValueError):
            int32(-9223372036854775809)
    except Exception as e:
        assert False, f"An error occured: {e}"


def test_int64_repr():
    try:
        value = 92233725808
        int_obj = int64(value)
        assert repr(int_obj) == f'int64({value})'
    except Exception as e:
        assert False, f"An error occurred: {e}"
