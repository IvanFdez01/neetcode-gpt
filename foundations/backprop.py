import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        x = np.array(x, dtype=np.float64)
        w = np.array(w, dtype=np.float64)
        preact = x @ w + b
        act = 1 / (1 + np.exp(-preact))
        loss = 0.5 * (act - y_true)**2
        dL_dw = []
        for x_i in x:
            dL_dw.append(np.round((act - y_true) * act * (1 - act) * x_i, 5))
        dL_db = np.round((act - y_true) * act * (1 - act), 5)
        return (dL_dw, dL_db)
