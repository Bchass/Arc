from . import data_types


class csr_matrix:

    def __init__(self, rows, cols, dtype=None, data=None, row=None):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.matrix = []

        if data is not None and row is not None:
            self.matrix = [[0] * self.cols for _ in range(self.rows)]

            for i in range(len(data)):
                row_index = i // self.cols
                col_index = row[i]
                self.matrix[row_index][col_index] = data[i]
                if col_index == data[i]:
                    raise IndexError("Row elements cannot match data elements")

        elif rows is not None and cols is not None and data is None:
            self.matrix = [[0] * rows for _ in range(cols)]
        else:
            self.matrix = []

    def __call__(self):
        return self

    def __str__(self):
        if self.dtype is not None:
            # dtype_str = self.dtype.__name__
            matrix_str = "\n".join(
                ["[" + ", ".join(map(str, row)) + "]" for row in self.matrix])
            # matrix_str = matrix_str + ", dtype=" + dtype_str
        else:
            matrix_str = "\n".join(
                ["[" + ", ".join(map(str, row)) + "]" for row in self.matrix])
        return str(matrix_str)

    def __repr__(self):
        if self.matrix is None:
            return "None"
        return str(self)
