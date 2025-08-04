import numpy as np
from forward import DenseLayer, Activation_ReLU, Activation_Softmax, Loss_CategoticalCrossentropy

np.random.seed(0)

# Dummy input data: 3 samples, 2 features
X = np.array([[1.0, 2.0],
              [0.5, -1.5],
              [-1.0, 2.5]])

# Ground truth labels (class indices for classification)
y = np.array([0, 1, 1])

# Create layers
dense1 = DenseLayer(2, 3)             # 2 inputs → 3 neurons
activation1 = Activation_ReLU()

dense2 = DenseLayer(3, 3)             # 3 inputs → 3 outputs (for 3 classes)
activation2 = Activation_Softmax()

# Forward pass through first dense layer + ReLU
dense1.forward(X)
print(dense1.output)
activation1.forward(dense1.output)
print(activation1.output)

# Forward pass through second dense layer + Softmax
dense2.forward(activation1.output)
print(dense2.output)
activation2.forward(dense2.output)

# Loss calculation
loss_function = Loss_CategoticalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

# Output everything
print("Predicted probabilities:\n", activation2.output)
print("Loss:", loss)
