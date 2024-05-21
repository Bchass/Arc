class dia_matrix:

    """
    A class representing a diagonal row (dia) matrix.
    """

    def __init__(self, size, shape=None, dtype=None, other=None):
        """
        Initialize the DIA matrix.

        Parameters:
        - size  (int): Size of the matrix.
        - shape (type, optional): Shape of the matirx.
        - dtype (type, optional): Data type of the elements in the matrix.
        - other (type, optional): Second matrix for subtraction, addition, etc.
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

    def __call__(self):
        return self

    def __str__(self):
        """
        Get a string representation of the matrix.

        Returns:
        - str: String representation of the matrix.
        """

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
        """
        Get the value at the specified index.

        Parameters:
        - index (tuple): Tuple representing the index (row, col).

        Returns:
        - Any: Value at the specifed index.

        Raises:
        - IndexError: If Index is out of bounds.
        - IndexError: If unsupported index type is passed.
        """
        if isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, slice):
                if row < 0 and row >= self.size:
                    raise IndexError("Index out of bounds")
            return self.matrix[row][col]
        raise IndexError("Unsupported index type")

    def __repr__(self):
        """
        Get a string representation of the matrix.

        Returns:
        - str: String representation of the matrix.
        """
        if self.matrix is None:
            return "None"
        return str(self)

    def set_element(self, row, col, value):
        """
        Set the element at the specified row and column in the matrix.

        Parameters:
        - row (int): The row index.
        - col (int): The column index.
        - value: The value to set at the specified position.

        Raises:
        - ValueError: If the specified row or column is out of range or if trying to set an element not on the diagonal.
        """

        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            raise ValueError("Index out of range")
            # support left to right and right to left
        if row == col or row + col == self.size - 1:
            self.matrix[row][col] = value
        else:
            raise ValueError("Can only set elements on diagonal")

    def get_shape(self):
        """
        Get the shape of the matrix.

        Returns:
        - tuple: A tuple representing the shape of the matrix.
        """
        return self.shape

    def get_size(self):
        """
        Get the size of the matrix.

        Returns:
        - int: The size of the matrix.
        """
        return self.size

    def multiply(self, other):
        """
        Multiply this matrix with another matrix.

        Parameters:
        - other (dia_matrix): The matrix to multiply with.

        Returns:
        - dia_matrix: The result of the multiplication.

        Raises:
        - ValueError: If the size of matrix A does not equal the size of matrix D.
        - ValueError: If the matrices are not compatible for multiplication.
        - ValueError: If instance is not a dia_matrix.
        """

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for multiplication")

        if self.size != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                "to the number of rows in the second matrix")

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    result.matrix[i][j] += self.matrix[i][k] * \
                        other.matrix[k][j]
        return result

    def add(self, other):
        """
        Add another matrix to this matrix.

        Parameters:
        - other (dia_matrix): The matrix to add.

        Returns:
        - dia_matrix: The result of the addition.

        Raises:
        - ValueError: If the matrices are not of the same size.
        - ValueError: If the size of matrix is A is not equal to the shape of matrix D.
        - ValueError: If instance is not a dia_matrix.
        """

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for addition")

        if self.size != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                "to the number of rows in the second matrix")

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                result.matrix[i][j] = self.matrix[i][j] + \
                    other.matrix[i][j]
        return result

    def subtract(self, other):
        """
        Subtract another matrix from this matrix.

        Parameters:
        - other (dia_matrix): The matrix to subtract.

        Returns:
        - dia_matrix: The result of the subtraction.

        Raises:
        - ValueError: If the size of matrix A does not equal the size of matrix D.
        - ValueError: If the matrices are not of the same size or if either matrix has negative diagonal elements.
        - ValueError: If instance is not a dia_matrix.
        """

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for subtraction")

        for i in range(self.size):
            if self.matrix[i][i] < 0 or other.matrix[i][i] < 0:
                raise ValueError(
                    "Positive definite matrix cannot"
                    "have negative diagonal elements")

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return result

    def divide(self, other):
        """
        Divide this matrix by another matrix.

        Parameters:
        - other (dia_matrix): The matrix to divide by.

        Returns:
        - dia_matrix: The result of the division.

        Raises:
        - ValueErrorL If the size of matrix A does not equal the size of matrix D.
        - ValueError: If the size of matrix A does not equal the shape of matrix D.
        - ValueError: If instance is not a dia_matrix.
        - ValueError: If the matrices are not compatible for division or if division by zero is encountered.
        """

        if self.size != other.size:
            raise ValueError(
                "Matrices must be of the same size for division")

        if self.size != other.shape:
            raise ValueError(
                "Number of columns in the first matrix does not equal"
                "to the number of rows in the second matrix")

        if not isinstance(other, dia_matrix):
            raise ValueError("Instance is not a dia_matrix")

        result = dia_matrix(self.size)
        result.matrix = [[0] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if other.matrix[i][j] == 0:
                    raise ValueError("Division by zeros encountered")
                result.matrix[i][j] = self.matrix[i][j] // other.matrix[i][j]
        return result

    def toarray(self):
        """
        Convert the matrix to a dense array.

        Returns:
        - list: Dense representation of the matrix.
        """

        array = self.matrix[:]
        if self.dtype is not None:
            dtype_str = self.dtype.__name__
            return [row[:] for row in array], dtype_str
        else:
            return [row[:] for row in array]
