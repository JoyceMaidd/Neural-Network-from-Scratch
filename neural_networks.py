import numpy as np

np.random.seed(0)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
         sample_loss = self.forward(output, y)
         data_loss = np.mean(sample_loss)
         return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, y_pred, y_true):
        """
        Computes the gradient of categorical cross-entropy loss with respect to the model's output (y_pred).

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted probabilities from the model. Shape: (n_samples, n_classes).
            Each row should sum to 1.
        y_true : np.ndarray
            True class labels. Can be:
                - One-hot encoded: Shape (n_samples, n_classes)
                - Integer class indices: Shape (n_samples,)

        Sets:
        -----
        self.dinputs : np.ndarray
            Gradient of the loss with respect to y_pred.
        """

        # Number of samples in the batch
        samples = len(y_pred)

        # If labels are one-dimensional (integer encoded), convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # Clip predictions to avoid division by zero and log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # For categorical cross-entropy: dL/dy_pred = -y_true / y_pred
        self.dinputs = -y_true / y_pred_clipped

        # Average gradient over all samples in the batch
        self.dinputs = self.dinputs / samples


# Combination of softmax activation and loss categorical crossentropy for LAST LAYER
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    #Forward pass
    def forward(self, input, y_true):
        self.activation.forward(input)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, y_pred, y_true):
        """
        Computes the gradient of loss with respect to inputs for the combined
        Softmax activation and Categorical Cross-Entropy loss.

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted probabilities after softmax.
            Shape: (n_samples, n_classes).
        y_true : np.ndarray
            True class labels.
            Shape:
                - (n_samples,) if integer-encoded
                - (n_samples, n_classes) if one-hot encoded

        Sets:
        -----
        self.dinputs : np.ndarray
            Gradient of the loss with respect to inputs to this layer.
        """

        # Number of samples in the batch
        samples = len(y_pred)

        # If labels are one-hot encoded, convert them to integer class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy the predictions so we can safely modify without changing original data
        self.dinputs = y_pred.copy()

        # Derivative simplification that comes from combining
        # softmax activation with categorical cross-entropy loss
        self.dinputs[range(samples), y_true] -= 1

        # Average gradient over all samples
        self.dinputs = self.dinputs / samples