import ctypes

'''
These are LEVEL 1 functions: https://www.netlib.org/blas/#_history
For implementation: https://developer.apple.com/documentation/accelerate
'''


class arc_BLAS:

    @staticmethod
    def sdot(x, y):

        # Load the Accelerate framework
        libblas = ctypes.CDLL(
            '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib')

        # Arguments and return types
        libblas.cblas_sdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        libblas.cblas_sdot.restype = ctypes.c_float

        n = len(x)
        # Convert Python lists to C arrays
        x_arr = (ctypes.c_float * n)(*x)
        y_arr = (ctypes.c_float * n)(*y)
        result = libblas.cblas_sdot(n, x_arr, 1, y_arr, 1)
        rounded = round(result, 2)
        return rounded

    @staticmethod
    def sdsdot(x, y):
        libblas = ctypes.CDLL(
            '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib')

        libblas.cblas_sdsdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

        libblas.cblas_sdsdot.restype = ctypes.c_double

        n = len(x)
        x_arr = (ctypes.c_double * n)(*x)
        y_arr = (ctypes.c_double * n)(*y)
        alpha = [a * b for a, b in zip(x, y)]
        result = libblas.cblas_sdsdot(ctypes.c_int(n), x_arr, 1, y_arr, 1) + sum(alpha)
        rounded = round(result, 2)
        return rounded

    @staticmethod
    def sasum(x):
        libblas = ctypes.CDLL(
            '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib')

        libblas.cblas_sdot.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_float), ctypes.c_int]

        libblas.cblas_sasum.restype = ctypes.c_float

        n = len(x)
        x_arr = (ctypes.c_float * n)(*x)
        result = libblas.cblas_sasum(n, x_arr, 1)
        return result
