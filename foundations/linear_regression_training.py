import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        X = np.array(X, dtype=np.float64)
        model_prediction = np.array(model_prediction, dtype=np.float64)
        ground_truth = np.array(ground_truth, dtype=np.float64)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = np.array(initial_weights, dtype=np.float64)
        for _ in range(num_iterations):
            x_pred = self.get_model_prediction(X, weights)
            for j in range(len(weights)):
                grad = self.get_derivative(x_pred, Y, len(x_pred), X, j)
                weights[j] -= self.learning_rate * grad
            print(x_pred)  
        return np.round(weights, 5) 

