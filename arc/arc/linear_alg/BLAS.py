import ctypes

class arc_BLAS:
    @staticmethod
    def dot_product(x, y):
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
        return result
