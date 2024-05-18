from . import data_types

# TODO: Anything 100x100 and it falls behind Scipy in the benchmark


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

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return self.matrix[row][col]
        else:
            raise TypeError("Indexing csr_matrix requires a tuple (row, col)")

    def __repr__(self):
        if self.matrix is None:
            return "None"
        return str(self)

    def get_rows(self):
        return self.rows

    def get_cols(self):
        return self.cols

    def get_size(self):
        return self.rows * self.cols

    def toarray(self):

        array = self.matrix[:]
        if self.dtype is not None:
            dtype_str = self.dtype.__name__
            return [row[:] for row in array], dtype_str
        else:
            return [row[:] for row in array]

    def nnz(self):

        count = 0
        for row in self.matrix:
            for element in row:
                if element != 0:
                    count += 1
        return count

    def multiply(self, other):

        result = csr_matrix(self.rows, self.cols)
        result.matrix = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    if other.matrix[k][j] == 0:
                        raise ValueError("Multiplication by zeros encountered")
                    result.matrix[i][j] += self.matrix[i][k] * \
                        other.matrix[k][j]
        return result
