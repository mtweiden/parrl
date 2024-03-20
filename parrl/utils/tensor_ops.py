from numpy import complex128
from numpy import float32
from numpy import ndarray
from numpy import stack


def split_complex_matrix(matrix: ndarray) -> ndarray:
    if matrix.dtype == complex128:
        return stack([matrix.real, matrix.imag], dtype=float32)
    else:
        return matrix