class csr_matrix:

    def __init__(self, rows, cols, dtype=None, data=None, row=None, col=None):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.matrix = [[0] * cols for _ in range(rows)]

        if data is not None:
            if row is not None and col is not None:
                if len(data) != len(row) or len(data) != len(col):
                    raise ValueError(
                        "Lengths of data, row, and col must match.")
                for d, r, c in zip(data, row, col):
                    if 0 <= r < rows and 0 <= c < cols:
                        self.matrix[r][c] = d
            else:
                if len(data) != rows * cols:
                    raise ValueError("Length of data must match rows * cols.")
                for i in range(len(data)):
                    r, c = divmod(i, cols)
                    if 0 <= r < rows and 0 <= c < cols:
                        if data[i] != 0:
                            self.matrix[r][c] = data[i]

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

    def add(self, other):

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                "to the number of rows in the second matrix"
            )

        result = csr_matrix(self.rows, self.cols)
        result.matrix = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result

    def subtract(self, other):

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                " to the number of rows in the second matrix")

        result = csr_matrix(self.rows, self.cols)
        result.matrix = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return result

    def divide(self, other):

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")

        result = csr_matrix(self.rows, self.cols)
        result.matrix = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                if other.matrix[i][j] == 0:
                    raise ValueError("Divison by zeros encountered")
                result.matrix[i][j] = self.matrix[i][j] // other.matrix[i][j]
        return result
