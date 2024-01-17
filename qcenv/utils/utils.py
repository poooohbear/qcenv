import numpy as np


def complex2real(complex_array):
    b = np.zeros(2 * len(complex_array))
    b[0::2] = complex_array.real
    b[1::2] = complex_array.imag
    return b.astype(np.float32)
