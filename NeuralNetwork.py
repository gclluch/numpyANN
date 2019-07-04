import numpy as np

### activation functions and their derivatives
def sigmoid(Z) :
    return 1 / (1 + np.exp(-Z))

def relu(Z) :
    return np.maximum(0, Z)

def d_sigmoid(Z) :
    gZ = sigmoid(Z)
    return gZ * (1 - gZ)

def d_relu(Z) :
    return np.where(Z > 0, 1, 0)

def tanh(Z) :
    return np.tanh(Z)

def d_tanh(Z) :
    return 1 - tanh(Z) ** 2

def activate(Z, activation) :
    if activation == 'relu' : return relu(Z)
    elif activation == 'sigmoid' : return sigmoid(Z)
    elif activation == 'tanh' : return tanh(Z)

def d_activate(Z, activation) :
    if activation == 'relu' : return d_relu(Z)
    elif activation == 'sigmoid' : return d_sigmoid(Z)
    elif activation == 'tanh' : return d_tanh(Z)


### Neural network architecture
np.random.seed(1)
class neural_network :

    #initialize network
    def __init__(self, layers, X, Y) :
        self.layers = layers
        self.cache = {}
        self.gradients = {}
        self.parameters = {}
        self.X = X
        self.Y = np.array(Y).reshape(1, Y.shape[0])
        layers[0] = (X.shape[0], None)
        self.L = len(layers) - 1

        #He initialization of parameters
        for i in range(1, len(layers) ) :
            cur_layer = layers[i][0]
            prev_layer = layers[i-1][0]
            self.parameters['W' + str(i)] = rng.randn(cur_layer, prev_layer) * np.sqrt(2 / prev_layer)
            self.parameters['b' + str(i)] = np.zeros((cur_layer, 1))


    def forward_propagation(self) :
        Z = self.parameters['W1'].dot(self.X) + self.parameters['b1']
        A = activate(Z, layers[1][1])
        self.cache['Z1'] = Z
        self.cache['A1'] = A
        for i in range(2, len(self.layers)) :
            A_prev = A
            Z = self.parameters['W' + str(i)].dot( A_prev ) + self.parameters['b' + str(i)]
            A = activate(Z, layers[i][1])
            self.cache['Z' + str(i)] = Z
            self.cache['A' + str(i)] = A


    def backward_propagation(self, lambd) :
        dA = 0
        A = self.cache['A' + str(self.L)]
        #initialize with binary cost
        if self.layers[self.L][1] == 'sigmoid' : dA = -( np.divide(self.Y, A) - np.divide(1-self.Y, 1-A) )
        for i in reversed(range(1, self.L + 1)) :
            A_prev = np.array([])
            m = self.X.shape[1]
            if i == 1 :
                A_prev = self.X
            else : A_prev = self.cache['A' + str(i-1)]
            #compute and store gradients
            dZ = dA * d_activate(self.cache['Z' + str(i)], layers[i][1])
            dA = np.dot(self.parameters['W' + str(i)].T, dZ)

            self.gradients['dW' + str(i)] = (1/m) * dZ.dot(A_prev.T) + (lambd/m)*self.parameters['W' + str(i)]
            self.gradients['db' + str(i)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)


    def update_parameters(self, alpha) :
        for i in range(1, self.L+1) :
            self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - alpha * self.gradients['dW' + str(i)]
            self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - alpha * self.gradients['db' + str(i)]


    #binary cross-entropy with regularization
    def cost(self, lambd) :
        A = self.cache['A' + str(self.L)]
        m = self.Y.shape[1]
        regSum = 0
        for i in range(1, len(self.layers) ) :
            regSum += np.sum(np.square(self.parameters['W' + str(i)]))

        cost = - (1/m) * ( np.dot(self.Y, A.T) + np.dot(1 - self.Y, 1 - A.T)) + (lambd/(2*m)) * regSum
        return np.squeeze(cost)


    def train(self, alpha, iterations, lambd, show_cost = False) :
        costs = []
        for i in range(iterations) :
            self.forward_propagation()
            cost = self.cost(lambd)
            costs.append(cost)
            if show_cost == True and i%1000 == 0 : print(cost)
            self.backward_propagation(lambd)
            self.update_parameters(alpha)

        if show_cost == True :
            plt.plot(np.arange(0,iterations), costs)
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.show()


    # predict on input data
    def predict(self, X_test) :
        Z = self.parameters['W1'].dot(X_test) + self.parameters['b1']
        A = activate(Z, layers[1][1])
        for i in range(2, len(self.layers)) :
            A_prev = A
            Z = self.parameters['W' + str(i)].dot( A_prev ) + self.parameters['b' + str(i)]
            A = activate(Z, layers[i][1])
        return np.where(A > .5, 1, 0)


### Use

# structure of neural net - (layer size, activation function)
layers = {1: (5, 'relu'), 2: (10, 'relu'), 3: (1, 'sigmoid')}

# initialize network
network = neural_network(layers, X_train, y_train)

# train network
network.train(alpha=.05, iterations=4000, lambd=100, show_cost=True)

# predict on trained network
y_hat = network.predict(X_valid)
