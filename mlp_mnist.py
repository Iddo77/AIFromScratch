import mnist
import os
import tqdm
import numpy as np

from dropout import Dropout
from init_weights import WeightInitType
from linear_layer import Linear
from loss import SoftmaxCrossEntropyLoss
from optimizer import GradientDescent, Momentum
from sequential_layer import Sequential
from activation import Tanh


class MLPMnist:
    def __init__(self, learning_rate=0.01, use_momentum=True):
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
        layers = [
            Linear(784, 30, WeightInitType.XAVIER_NORMAL),
            self.dropout1,
            Tanh(),
            Linear(30, 10, WeightInitType.XAVIER_NORMAL),
            self.dropout2,
            Tanh(),
            Linear(10, 10, WeightInitType.XAVIER_NORMAL)
        ]
        self.model = Sequential(layers)
        self.outputs: np.ndarray | None = None
        self.loss = SoftmaxCrossEntropyLoss()
        if use_momentum:
            self.optimizer = Momentum(learning_rate, momentum=0.99)
        else:
            self.optimizer = GradientDescent(learning_rate)

        self.mnist_folder = os.path.join(os.path.dirname(__file__), 'mnist')
        os.makedirs(self.mnist_folder, exist_ok=True)
        mnist.temporary_dir = lambda: self.mnist_folder
        self.train_images = mnist.train_images()
        self.train_labels = mnist.train_labels()
        self.test_images = mnist.test_images()
        self.test_labels = mnist.test_labels()

        self.preprocess_images()
        self.one_hot_encode_labels()

    def preprocess_images(self):
        # divide by 256 to get a value between 0 and 1
        self.train_images = self.train_images / 256

        # subtract average so that the values are 0 on average
        avg = np.mean(self.train_images)
        self.train_images -= avg

        # now do the same for test images
        self.test_images = self.test_images / 256 - avg

        # flatten last 2 dimensions
        self.train_images = self.train_images.reshape(self.train_images.shape[0], 1, -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], 1, -1)

    def one_hot_encode_labels(self):
        one_hot_matrix = np.eye(10)  # 10 digits
        self.train_labels = [one_hot_matrix[i] for i in self.train_labels]
        self.test_labels = [one_hot_matrix[i] for i in self.test_labels]

    def forward(self, inputs) -> np.ndarray:
        self.outputs = self.model.forward(inputs)
        return self.outputs

    def backward(self, Y: np.ndarray):
        error = self.loss.gradient(Y, self.outputs)
        self.model.backward(error)
        self.optimizer.step(self.model)

    def train(self, epochs=1):
        correct = 0
        total_loss = 0.0

        self.dropout1.train = True
        self.dropout2.train = True

        for epoch in range(epochs):
            with tqdm.trange(len(self.train_images)) as t:
                for i in t:
                    y_true = self.train_labels[i]
                    y_pred = self.forward(self.train_images[i])
                    if np.argmax(y_pred) == np.argmax(y_true):
                        correct += 1
                    total_loss += self.loss.loss(y_true, y_pred)
                    self.backward(y_true)

                    avg_loss = total_loss / (i + 1)
                    acc = correct / (i + 1)
                    t.set_description(f"Epoch: {epoch}  loss: {avg_loss:.3f}  accuracy: {acc:.3f}")

    def validate(self):
        correct = 0
        total_loss = 0.0

        self.dropout1.train = False
        self.dropout2.train = False

        with tqdm.trange(len(self.test_images)) as t:
            for i in t:
                y_true = self.test_labels[i]
                y_pred = self.forward(self.test_images[i])
                if np.argmax(y_pred) == np.argmax(y_true):
                    correct += 1
                total_loss += self.loss.loss(y_true, y_pred)

                avg_loss = total_loss / (i + 1)
                acc = correct / (i + 1)
                t.set_description(f"Loss: {avg_loss:.3f}  accuracy: {acc:.3f}")


if __name__ == "__main__":
    nn = MLPMnist(use_momentum=True)
    nn.train()
    nn.model.save_weights('mnist_weights.json')
    # nn.model.load_weights('mnist_weights.json')
    # nn.validate()
