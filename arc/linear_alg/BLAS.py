import ctypes
from ctypes import POINTER, c_int, c_float

'''
These are LEVEL 1 Single functions: https://www.netlib.org/blas/#_history
For implementation: https://developer.apple.com/documentation/accelerate
'''


class arc_BLAS:

    def load_BLAS():

        # Load the Accelerate framework
        return ctypes.CDLL(
            '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib')  # noqa

    @staticmethod
    def sdot(x, y):

        blas_lib = arc_BLAS.load_BLAS()

        # Arguments and return types
        blas_lib.cblas_sdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int]

        blas_lib.cblas_sdot.restype = ctypes.c_float

        n = len(x)
        # Convert Python lists to C arrays
        x_arr = (ctypes.c_float * n)(*x)
        y_arr = (ctypes.c_float * n)(*y)
        result = blas_lib.cblas_sdot(n, x_arr, 1, y_arr, 1)
        rounded = round(result, 2)
        return rounded

    @staticmethod
    def sdsdot(x, y):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_sdsdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ctypes.c_int]

        blas_lib.cblas_sdsdot.restype = ctypes.c_double

        n = len(x)
        x_arr = (ctypes.c_double * n)(*x)
        y_arr = (ctypes.c_double * n)(*y)
        alpha = [a * b for a, b in zip(x, y)]
        result = blas_lib.cblas_sdsdot(
            ctypes.c_int(n), x_arr, 1, y_arr, 1) + sum(alpha)
        rounded = round(result, 2)
        return rounded

    @staticmethod
    def sasum(x):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_sdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_float), ctypes.c_int]

        blas_lib.cblas_sasum.restype = ctypes.c_float

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        result = blas_lib.cblas_sasum(n, x_arr, 1)
        return result

    @staticmethod
    def sscal(x):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_sscal.argtypes = [
            c_int, POINTER(c_float), c_int, c_float]
        blas_lib.cblas_sscal.restype = None

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        alpha = ctypes.c_float(2.0)
        blas_lib.cblas_sscal(n, x_arr, 1, alpha)
        return list(x_arr)

    @staticmethod
    def scopy(x, y):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_scopy.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(
                ctypes.c_float), ctypes.c_int
        ]

        blas_lib.cblas_scopy.restype = None

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        y_arr = (ctypes.c_float * n)()
        blas_lib.cblas_scopy(n, x_arr, 1, y_arr, 1)
        return x_arr

    @staticmethod
    def sasum(x):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_sasum.argtypes = [
            c_int, POINTER(c_float), c_int]
        blas_lib.cblas_sasum.restype = c_float

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        result = blas_lib.cblas_sasum(n, x_arr, 1)
        return result

    @staticmethod
    def sswap(x, y):

        blas_lib = arc_BLAS.load_BLAS()

        blas_lib.cblas_sswap.argtypes = [c_int, POINTER(
            c_float), c_int, POINTER(c_float), c_int]

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        y_arr = (ctypes.c_float * n)(*y)
        blas_lib.cblas_sswap(n, x_arr, 1, y_arr, 1)

        for i in range(n):
            x[i] = x_arr[i]
            y[i] = y_arr[i]

        return x, y
