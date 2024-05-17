from . import data_types


class csr_matrix:

    def __init__(self, rows, cols, dtype=None, data=None, row=None, col=None):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.matrix = []

        if data is not None and row is not None and col is not None:
            self.matrix = [[0] * self.cols for _ in range(self.rows)]

            for i in range(len(data)):
                row_index = row[i]
                col_index = col[i]
                self.matrix[row_index][col_index] = data[i]

        elif data is not None and row is None and col is None:
            self.matrix = [[0] * self.cols for _ in range(self.rows)]

            for i in range(len(data)):
                row_index = i // self.cols
                col_index = i % self.cols
                self.matrix[row_index][col_index] = data[i]

        elif rows is not None and cols is not None and data is None:
            self.matrix = [[0] * cols for _ in range(rows)]
        else:
            self.matrix = []

    def __call__(self):
        return self

    def __str__(self):
        if self.dtype is not None:
            dtype_str = self.dtype.__name__
            matrix_str = "\n".join(
                ["[" + ", ".join(map(str, row)) + "]" for row in self.matrix])
            matrix_str = matrix_str + ", dtype=" + dtype_str
        else:
            matrix_str = "\n".join(
                ["[" + ", ".join(map(str, row)) + "]" for row in self.matrix])
        return str(matrix_str)

    def __repr__(self):
        if self.matrix is None:
            return "None"
        return str(self)

    def toarray(self):

        array = self.matrix[:]
        if self.dtype is not None:
            dtype_str = self.dtype.__name__
            return [row[:] for row in array], dtype_str
        else:
            return [row[:] for row in array]
