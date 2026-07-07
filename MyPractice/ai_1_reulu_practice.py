import numpy as np

def squash(z):
    return np.maximum(0, z) #this will return the maximum value between 0 and z, effectively implementing the ReLU activation function where either positove or 0(instead of negative values) will be returned

W1 = np.array([[0.5, -0.3], [-0.2, 0.8]]) #weights for layer 1, each row represents a neuron and each column represents an input feature
X1 = np.array([1, 0]) #input to the layer, a 2D vector representing two features
b1 = np.array([0.1, -0.2]) #biases for layer 1, one bias per neuron

#note: the @ operator is used for matrix multiplication in numpy, which is equivalent to np.dot() function
layer1_output_without_squash = W1 @ X1 + b1 #matrix multiplication of weights and input, then adding biases
layer1_output = squash(layer1_output_without_squash) #applying the ReLU

w2 = np.array([[0.6, -0.4]]) #weights for layer 2, one neuron with two inputs from layer 1
b2 = np.array([0.05]) #bias for layer 2, one bias
layer2_output_without_squash = w2 @ layer1_output + b2 #matrix multiplication of weights and layer 1 output, then adding bias
layer2_output = squash(layer2_output_without_squash) #applying the ReLU

print("Layer 1 output without squash:", layer1_output_without_squash)
print("Layer 1 output after squash:", layer1_output)
print("Layer 2 output without squash:", layer2_output_without_squash)
print("Layer 2 output after squash:", layer2_output)