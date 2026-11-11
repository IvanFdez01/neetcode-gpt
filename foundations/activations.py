import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z = np.asarray(z, dtype=np.float64)
        return np.round(0.5 * np.tanh(0.5 * z) + 0.5, 5)
         

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        z = np.array(z, dtype=np.float64)
        return np.maximum(0, z)
