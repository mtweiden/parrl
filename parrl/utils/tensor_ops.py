from numpy import complex128
from numpy import float64
from numpy import float32
from numpy import ndarray
from numpy import stack

from torch import Tensor


def split_complex_matrix(matrix: ndarray) -> ndarray:
    if matrix.dtype == complex128:
        return stack([matrix.real, matrix.imag], dtype=float32)
    elif matrix.dtype == float64:
        return matrix.astype(float32)
    else:
        return matrix


def convert_array_to_tensor(
    array: ndarray,
    device: str = 'cpu',
) -> Tensor:
    array = split_complex_matrix(array)
    array = Tensor(array).float()
    if device == 'cpu':
        array = array.cpu()
    return array
