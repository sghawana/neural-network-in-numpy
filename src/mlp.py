import numpy as np

class Linear():
    def __init__(self, input_size, output_size, activation = 'sigmoid', use_bias=True, dropout=0, use_act=True, eval=True, regularisation='None', beta=0):
        self.in_size = input_size
        self.out_size = output_size
        self.activation = activation
        self.use_act = use_act
        self.use_bias = use_bias
        self.eval = eval
        self.reg = regularisation
        self.beta = beta
        self.dropout = dropout

        self.weight = np.random.uniform(-1, 1, size=(self.out_size, self.in_size))
        if self.use_bias:
            self.bias = np.random.uniform(-1, 1, size=(self.out_size, 1))
        else:
            self.bias = None

        self.prev_a = 0
        self.z = 0
        self.a = 0
        self.delta = 0

    def activate(self,x):
      if self.activation ==  'relu':
          return np.maximum(0, x)
      elif self.activation == 'sigmoid':
          return 1 / (1 + np.exp(-x))

    def forward(self, x):
        if self.eval:
          self.prev_a = x
        x = self.weight @ x
        if self.bias is not None:
            bias_adjusted = np.tile(self.bias, (1, x.shape[1]))
            x += bias_adjusted
        x = self.dropout_layer(x)
        if self.eval:
          self.z = x
        if self.use_act == True:
            x = self.activate(x)
        if self.eval:
          self.a = x
        return x

    def update_delta(self, next_delta, next_weight):
      if self.activation == 'sigmoid':
          if self.use_act:
              sigma = 1 / (1 + np.exp(-self.z))
              one_v = np.ones_like(self.z)
              temp = sigma * (one_v - sigma)
              self.delta = temp * (next_weight.T@next_delta)
          else:
              self.delta = (next_weight.T@next_delta)

      elif self.activation == 'relu':
          if self.use_act:
              temp = np.where(self.z >= 0, 1, 0)
              self.delta = temp * (next_weight.T@next_delta)
          else:
              self.delta = (next_weight.T@next_delta)
      return self.delta

    def get_parameters(self):
        return {'weight': self.weight, 'bias': self.bias} if self.use_bias else {'weight': self.weight}

    def update_layer_parameters(self, lr):
        self.weight = self.weight - lr * np.clip(self.delta, -10000, 10000) @ self.prev_a.T
        self.bias = self.bias - lr * np.clip(self.delta, -10000, 10000)
        if self.reg == 'L2' :
          self.weight = (1- lr*self.beta)*self.weight
          self.bias = (1- lr*self.beta)*self.bias
        elif self.reg == 'L1' :
          self.weight = self.weight - lr*self.beta * np.sign(self.weight)
          self.bias = self.bias - lr*self.beta * np.sign(self.bias)

    def clear_grad(self):
        self.layer_grad = 0

    def dropout_layer(self, x):
        result = x.copy()
        for i in range(x.shape[1]):
            num_zeros = int(self.beta * x.shape[0])
            zero_indices = np.random.choice(x.shape[0], size=num_zeros, replace=False)
            result[zero_indices, i] = 0
        return result



class MLP():
    def __init__(self, input_size, output_size, hid_sizes=[], activation='sigmoid', is_softmax=False, reg='None', b=0, layer_dropout=0):
        self.in_size = input_size
        self.out_size = output_size
        self.activation = activation
        self.is_softmax = is_softmax
        self.layer_sizes = [input_size] + hid_sizes
        self.layer_list = []
        self.reg = reg
        self.beta = b
        self.layer_dropout = layer_dropout

        for i in range(1, len(self.layer_sizes)):
          self.layer_list.append(Linear(self.layer_sizes[i-1], self.layer_sizes[i], self.activation, regularisation=reg, beta=b, dropout=self.layer_dropout))
        self.output_layer = Linear(self.layer_sizes[-1], self.out_size, self.activation, use_act=False, regularisation=reg, beta=b)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_pass(self, x):
        for layer in self.layer_list:
            x = layer.forward(x)
        x = self.output_layer.forward(x)
        if self.is_softmax:
            x = self.softmax(x)
        return x

    def weights(self):
        weights = []
        for layer in self.layer_list:
            weights.append(layer.get_weight())
        weights.append(self.output_layer.get_weight())
        return weights

    def biases(self):
        biases = []
        for layer in self.layer_list:
            biases.append(layer.get_bias())
        biases.append(self.output_layer.get_bias())
        return biases

    def parameters(self):
        return {'weights': self.weights(), 'biases': self.biases()}