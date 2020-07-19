import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.ioff()
matplotlib.use('Qt5Agg')

class Brain:
    def __init__(self, in_size, out_size, params = None):
        self.hidden_neurons = 8
        if params is None:
            self.layer = np.random.ranf((in_size,self.hidden_neurons))-0.5
            # s = np.sum(self.layer)
            # self.layer = self.layer/s

            self.bias = np.random.ranf((self.hidden_neurons))-0.5
            # s = np.sum(self.layer)
            # self.bias = self.bias/s

            self.hidden_layer = np.random.ranf((self.hidden_neurons,self.hidden_neurons))-0.5
            # s = np.sum(self.hidden_layer)
            # self.hidden_layer = self.hidden_layer / s

            self.hidden_layer_2 = np.random.ranf((self.hidden_neurons, out_size)) - 0.5
            # s = np.sum(self.hidden_layer_2)
            # self.hidden_layer_2 = self.hidden_layer_2 / s
        else:
            self.layer = params[0]
            self.hidden_layer = params[1]
            self.bias = params[2]
            self.hidden_layer_2 = params[3]
    def print(self):
        print("layer one: ")
        print(self.layer)
        print("bias (layer one): ")
        print(self.bias)
        print("hidden layer: ")
        print(self.hidden_layer)
        print("hidden layer 2: ")
        print(self.hidden_layer_2)

    def predict(self, input):
        hidden =  np.maximum(np.matmul(input, self.layer) + self.bias,0.0)
        hidden = np.minimum(np.maximum(np.matmul(hidden, self.hidden_layer),-1.0),1.0)
        out = np.matmul(hidden, self.hidden_layer_2)
        return out
    def get_params(self):
        return [self.layer, self.hidden_layer, self.bias, self.hidden_layer_2]
    def mutate(self, lr = 0.7):
        i = np.random.randint(self.layer.shape[0])
        j = np.random.randint(self.layer.shape[1])
        self.layer[i][j] += (np.random.ranf(1)-0.5) *lr
        #self.layer[i][j] = 0.0
        i = np.random.randint(self.hidden_layer.shape[0])
        j = np.random.randint(self.hidden_layer.shape[1])
        self.hidden_layer[i][j] += (np.random.ranf(1) - 0.5) * lr
        #self.hidden_layer[i][j] =0.0
        i = np.random.randint(self.hidden_layer_2.shape[0])
        j = np.random.randint(self.hidden_layer_2.shape[1])
        self.hidden_layer_2[i][j] += (np.random.ranf(1) - 0.5) * lr
        #self.hidden_layer_2[i][j] = 0.0
