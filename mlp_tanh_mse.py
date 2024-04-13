import numpy as np

from init_weights import WeightInitType
from linear_layer import Linear
from loss import MSELoss
from optimizer import GradientDescent, Momentum
from sequential_layer import Sequential
from activation import Relu, Tanh


class MLPTanhMse:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, use_momentum=False):
        layers = [
            # Kaiming-He initialization is designed for layers with Relu activation
            Linear(input_size, hidden_size, WeightInitType.KAIMING_HE),
            Relu(),
            # Xavier initialization is for layers with activation functions that are symmetric around zero, like sigmoid
            Linear(hidden_size, output_size, WeightInitType.XAVIER_NORMAL),
            Tanh()
        ]
        self.model = Sequential(layers)
        self.outputs: np.ndarray | None = None
        self.loss = MSELoss()  # BCE loss is not possible with Tanh
        if use_momentum:
            self.optimizer = Momentum(learning_rate)
        else:
            self.optimizer = GradientDescent(learning_rate)

    def forward(self, inputs) -> np.ndarray:
        self.outputs = self.model.forward(inputs)
        return self.outputs

    def backward(self, Y: np.ndarray):
        error = self.loss.gradient(Y, self.outputs)
        self.model.backward(error)
        self.optimizer.step(self.model)

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(Y)
            if epoch % 100 == 0:
                loss = self.loss.loss(Y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

        loss = self.loss.loss(Y, self.forward(X))
        print(f"Final Loss: {loss}")


if __name__ == "__main__":
    # Example: XOR function
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    layers = []

    nn = MLPTanhMse(input_size=2, hidden_size=6, output_size=1, use_momentum=True)
    nn.train(X, Y)
