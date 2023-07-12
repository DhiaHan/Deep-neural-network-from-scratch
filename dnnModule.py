import numpy as np
class DeepNeuralNetwork:
    def __init__(self, layers, activations_names, loss):
        assert(len(layers) == len(activations))
        self.layers = layers
        self.relu = lambda x: (x > 0) * x
        self.relu_derivative = lambda x: (x > 0) * 1
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative = lambda x: self.sigmoid(x) * (1 - self.sigmoid(x))
        self.tanh = lambda x: np.tanh(x)
        self.tanh_derivative = lambda x: 1 - np.power(np.tanh(x), 2)
        self.linear = lambda x: x
        self.linear_derivative = lambda x: np.ones(x.shape)
        self.softmax = lambda x: np.exp(x) / sum(np.exp(x))
        self.activations, self.derivatives = self.__string_to_function(activations_names)
        mean_squared_error = lambda Y_pred, Y, m: (1/m) * ((Y - Y_pred) ** 2).sum()
        binary_cross_entropy = lambda Y_pred, Y, m: (-1/m) * (Y*np.log(Y_pred) + (1-Y)*np.log(1-Y_pred)).sum()
        self.loss = eval(loss)
        self.weights = None
        self.L = len(layers)
        self.losses = []
    
    def __string_to_function(self, activations_names):
        activations_ = []
        derivatives_ = {}
        for i, act_name in enumerate(activations_names):
            activations_.append(eval('self.' + act_name))
            derivatives_["d" + str(i + 1)] = eval('self.' + act_name + '_derivative')
        derivatives_.popitem()
        return activations_, derivatives_
    
    def __initialize_weights(self, input_size):
        weights = {}
        L = len(self.layers)
        weights["W1"] = np.random.randn(self.layers[0], input_size)
        weights["b1"] = np.zeros((self.layers[0], 1))
        for l in range(1, L):
            weights["W" + str(l + 1)] = np.random.randn(self.layers[l], self.layers[l - 1])
            weights["b" + str(l + 1)] = np.zeros((self.layers[l], 1))
        return weights
    
    def __forward_propagation(self, X):
        cache = {}
        m = X.shape[1]
        cache["Z1"] = np.dot(self.weights["W1"], X) + self.weights["b1"]
        cache["A1"] = self.activations[0](cache["Z1"])
        for l in range(1, self.L):
            next_W = self.weights["W" + str(l + 1)]
            next_b = self.weights["b" + str(l + 1)]
            prev_A = cache["A" + str(l)]
            next_Z = np.dot(next_W, prev_A) + next_b
            next_A = self.activations[l](next_Z)
            cache["Z" + str(l + 1)] = next_Z
            cache["A" + str(l + 1)] = next_A
        AL = cache["A" + str(self.L)]
        return AL, cache
    
    def __backward_propagation(self, X, Y, cache):
        gradients = {}
        m = X.shape[1]
        dZL = cache["A" + str(self.L)] - Y
        dWL = (1/m) * np.dot(dZL, cache["A" + str(self.L - 1)].T)
        dbL = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        gradients["dZ" + str(self.L)] = dZL
        gradients["dW" + str(self.L)] = dWL
        gradients["db" + str(self.L)] = dbL
        for l in reversed(range(2, self.L + 1)):
            current_W = self.weights["W" + str(l)]
            current_dZ = gradients["dZ" + str(l)]
            prev_Z = cache["Z" + str(l - 1)]
            if l == 2:
                prev_prev_A = X
            else : prev_prev_A = cache["A" + str(l - 2)]
            prev_dZ = np.dot(current_W.T, current_dZ) * self.derivatives["d" + str(l - 1)](prev_Z)
            prev_dW = np.dot(prev_dZ, prev_prev_A.T)
            prev_db = (1/m) * np.sum(prev_dZ)
            gradients["dZ" + str(l - 1)] = prev_dZ
            gradients["dW" + str(l - 1)] = prev_dW
            gradients["db" + str(l - 1)] = prev_db
        return gradients
    
    def __update_weights(self, gradients, learning_rate):
        for l in range(1, self.L + 1):
            self.weights["W" + str(l)] -= learning_rate * gradients["dW" + str(l)]
            self.weights["b" + str(l)] -= learning_rate * gradients["db" + str(l)]
    
    def train(self, X, Y, epochs, learning_rate, print_each=100):
        m = X.shape[1] # X is a (features, m) array
        input_size = X.shape[0] # (features)
        self.weights = self.__initialize_weights(input_size)
        for i in range(epochs):
            AL, cache = self.__forward_propagation(X)
            cost = self.loss(AL, Y, m)
            self.lossed.append(cost)
            gradients = self.__backward_propagation(X, Y, cache)
            self.__update_weights(gradients, learning_rate)
            if (i % print_each == 0) : print("Epoch", i, "Loss", cost)
                
    def predict(self, X):
        Y_pre, cache = self.__forward_propagation(X)
        return Y_pre.T
