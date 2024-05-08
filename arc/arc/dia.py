from . import data_types


class dig_matrix:

    def __init__(self, size, shape=None, dtype=None, other=None):
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.other = other
        if dtype is None:
            if shape is None:
                self.matrix = [[0] * size for _ in range(size)]
            else:
                self.matrix = [[0] * shape for _ in range(size)]
        else:
            self.matrix = [[0] * shape for _ in range(size)]

    def __call__(self):
        return self

    # TODO: figure out how to print with dig_matrix(5,3,dtpye=Int8) directly
    def __str__(self):

        if self.dtype is not None:
            if isinstance(self.dtype, type):
                dtype_str = self.dtype.__name__
            else:
                dtype_str = str(self.dtype)
            return "\n".join(
                "[" + ", ".join(str(elem) for elem in row) + "]"for row in self.matrix
            ) + " , dtype=" + dtype_str
        else:
            return "\n".join(
                "[" + ", ".join(str(elem) for elem in row) + "]"for row in self.matrix
            )

    def set_element(self, row, col, value):

        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            raise ValueError("Index out of range")
            # support left to right and right to left
        if row == col or row + col == self.size - 1:
            self.matrix[row][col] = value
        else:
            raise ValueError("Can only set elements on diagonal")

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, slice):
                if row < 0 and row >= self.size:
                    raise IndexError("Index out of bounds")
            return self.matrix[row][col]
        raise IndexError("Unsupported index type")

    def multiply(self, other):

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for multiplication")

        result = dig_matrix(self.size)
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    result.matrix[i][j] += self.matrix[i][k] * \
                        other.matrix[k][j]
        return result
