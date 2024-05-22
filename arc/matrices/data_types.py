import ctypes

# TODO: Add more data types.


class int8(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int8)]

    def __init__(self, value=None):
        if not (-128 <= value <= 127):
            raise ValueError("Out of bounds")
        self.value = value

    def __repr__(self):
        return f"int8({self.value})"


class int16(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int16)]

    def __init__(self, value=None):
        if not (-32768 <= value <= 32767):
            raise ValueError("Out of bounds")
        self.value = value

    def __repr__(self):
        return f"int16({self.value})"


class int32(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32)]

    def __init__(self, value=None):
        if not (-2147483648 <= value <= 2147483647):
            raise ValueError("Out of bounds")
        self.value = value

    def __repr__(self):
        return f"int32({self.value})"


class int64(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int64)]

    def __init__(self, value=None):
        if not (-9223372036854775808 <= value <= 9223372036854775807):
            raise ValueError("Out of bounds")
        self.value = value

    def __repr__(self):
        return f"int64({self.value})"
