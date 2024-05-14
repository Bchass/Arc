
class csr_matrix:

    def __init__(self, rows=None, cols=None):
        self.rows = rows
        self.cols = cols

        if rows or cols is not None:
            self.matrix = [[0] * rows] * cols
        else:
            self.matrix = [[0] * rows for _ in range(cols)]

    def __call__(self):
        return self

    def __str__(self):
        matrix_str = "\n".join(
            ["[" + ", ".join(map(str, row)) + "]" for row in self.matrix])

        return matrix_str
