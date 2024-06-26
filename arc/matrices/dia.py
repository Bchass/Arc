from tinynumpy import tinynumpy as tnp

class dia_matrix:

    """
    A class representing a diagonal row (dia) matrix.
    """

    def __init__(self, size, shape=None, dtype=None, other=None, data=None, offsets=None):
        """
        Initialize the DIA matrix.

        Parameters:
        - size  (int): Size of the matrix.
        - shape (type, optional): Shape of the matirx.
        - dtype (type, optional): Data type of the elements in the matrix.
        - other (type, optional): Second matrix for subtraction, addition, etc.
        - data  (type, optional): Data to be inserted on the main diagonal or anti-diagonal
        """

        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.other = other

        if shape is None:
            shape = size
        else:
            if dtype is not None:
                self.matrix = [[0] * shape] * size
            else:
                self.matrix = [[0] * shape for _ in range(size)]

        if data is not None:
            for i in range(min(size, len(data))):
                self.matrix[i][i] = data[i]  # main diagonal
            if len(data) > size:
                for i in range(min(size, len(data))):
                    # anti-diagonal
                    self.matrix[size - 1 - i][i] = data[len(data) - 1 - i]

        if offsets is not None and data is not None:
            # make sure offsets are lined up with data
            for offset, d in zip(offsets, data):
                # check negative offsets if they are less than size or greater than or equal to 0
                if -size < offset < size:
                    if offset >= 0:
                        # handle positive offset
                        for i in range(size - offset):
                            self.matrix[i + offset][i] = d
                    else:
                        # handle negative offset
                        for i in range(size + offset):
                            self.matrix[i][i - offset] = d
                else:
                    print(f"Ignoring offset {offset}, it's out of bounds")

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

        return matrix_str

    def __getitem__(self, index):

        if isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, slice):
                if row < 0 and row >= self.size:
                    raise IndexError("Index out of bounds")
            return self.matrix[row][col]
        raise IndexError("Unsupported index type")

    def __repr__(self):

        if not any(self.matrix):
            return ""
        return str(self)

    def get_shape(self):

        return self.shape

    def get_size(self):

        return self.size

    def multiply(self, other):

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for multiplication")

        if self.shape != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal to the number of rows in the second matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    result.matrix[i][j] += self.matrix[i][k] * \
                        other.matrix[k][j]
        return result

    def add(self, other):

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for addition")

        if self.shape != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal to the number of rows in the second matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                result.matrix[i][j] = self.matrix[i][j] + \
                    other.matrix[i][j]
        return result

    def subtract(self, other):

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for subtraction")

        for i in range(self.size):
            if self.matrix[i][i] < 0 or other.matrix[i][i] < 0:
                raise ValueError(
                    "Positive definite matrix cannot have negative diagonal elements")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return result

    def divide(self, other):

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for division")

        if self.size != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                "to the number of rows in the second matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if other.matrix[i][j] == 0:
                    raise ValueError("Division by zeros encountered")
                result.matrix[i][j] = self.matrix[i][j] // other.matrix[i][j]
        return result

    def toarray(self):

        arr = self.matrix[:]
        return tnp.array(arr)
