import numpy as np


class Loss:
    def loss(self, y_true, y_pred):
        """ Returns an estimation of how wrong the predictions are. """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """ Returns the derivative of the loss. """
        raise NotImplementedError


class MSELoss(Loss):
    """
    Mean Squared Error loss
    """
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        # 1. apply power rule, which gives 2 * (y_true - y_pred)
        # 2. differentiate the inner function (y_true - y_pred) with respect to y_pred which gives -1
        # 3. combine by applying the chain rule, and you get -2 * (y_true - y_pred)
        return -2 * (y_true - y_pred)


class BCELoss(Loss):
    """
    Binary Cross Entropy loss
    """
    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax with Cross Entropy Loss
    Combines softmax activation with cross entropy loss for multi-class classification tasks.
    """
    @staticmethod
    def softmax(y_pred):
        # Normalize first by dividing by the largest value in y_pred. This is a stability trick to prevent overflow.
        m = np.max(y_pred, axis=1, keepdims=True)
        exps = np.exp(y_pred - m)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def loss(self, y_true, y_pred):
        # softmax outputs represent probabilities (likelihoods) for each class.
        p = self.softmax(y_pred)
        log_likelihood = -np.log(p + np.finfo(float).eps)  # Adding eps to avoid log(0) which would result in -inf

        # Negative log likelihood is used here because we want to minimize the loss during training:
        # Maximizing the likelihood of the correct class is equivalent to minimizing the negative log likelihood.
        # This conversion to minimization is useful because optimization algorithms typically minimize functions.
        negative_log_likelihood = -log_likelihood

        # Calculate the average negative log likelihood across all samples as the final cross-entropy loss.
        # This is equivalent to taking the mean of sums across the categorical labels,
        # where y_true is typically one-hot encoded.
        cross_entropy_loss = np.mean(np.sum(y_true * negative_log_likelihood, axis=1))
        return cross_entropy_loss

    def gradient(self, y_true, y_pred):
        p = self.softmax(y_pred)

        # 1. The partial derivative of log(p) with respect to p is 1/p.
        # 2. The partial derivative of -log(p) with respect to p is -1/p.
        # 3. The partial derivative with respect to y_pred is p_i*(1 - p_i) when p_i is the true class
        #    (because of one hot encoding)
        # 4. When p_i is not the true class (i.e., for all other classes where y_true_i = 0),
        #    the partial derivative of p_i with respect to y_pred_k (for k ≠ i) is -p_i * p_k.
        # 5. Combining these effects using the chain rule, for the softmax output p_i:
        #    - The derivative of the loss with respect to y_pred_i for the true class (where y_true_i = 1)
        #      is -1 * (1 - p_i), simplifying to (p_i - 1).
        #    - For all incorrect classes (where y_true_i = 0),
        #      the derivative of the loss with respect to y_pred_k is p_k.
        #    - Therefore, the complete gradient vector for the loss with respect to y_pred is
        #       the difference between the predicted probabilities p and the one-hot encoded true labels y_true,
        #       resulting in the vector (p - y_true).

        return p - y_true
