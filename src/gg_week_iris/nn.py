import numpy as np

class SimpleNeuralNet:
    # initialization function
    def __init__(
        self, input_size, hidden_size, output_size, learning_rate=0.0002, 
        activation='relu', init_biases_as_zero=True, dropout_p=0, temp=1,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.biases_as_zero = init_biases_as_zero
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        if self.biases_as_zero:
            self.bias_hidden = np.zeros((1, hidden_size))
            print(self.bias_hidden.shape)
            self.bias_output = np.zeros((1, output_size))
        else:
            self.bias_hidden = np.array(np.random.randn(1, hidden_size))
            self.bias_output = np.array(np.random.randn(1, output_size))
            
    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    # sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # relu function
    def relu(self, x):
        return np.maximum(0, x)
    # relu derivative
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    # softmax function
    def softmax(self, x, temp=1):
        exp_x = np.exp(x/temp - np.max(x/temp, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # forward function
    def forward(self, X, is_training=True, dropout_p=0):
        # from input layer to hidden input
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # hidden activation
        if self.activation == 'sigmoid':
            self.hidden_output = self.sigmoid(self.hidden_input)
        elif self.activation == 'relu':
            self.hidden_output = self.relu(self.hidden_input)
        else:
            raise ValueError("Invalid activation function.")
            
        # Apply dropout during training
        if is_training and dropout_p > 0:
            self.dropout_mask = np.random.binomial(1, dropout_p, size=self.hidden_output.shape) / dropout_p
            self.hidden_output *= self.dropout_mask
        
        # from hidden to output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # output activation
        self.output = self.softmax(self.output_input)
        return self.output
    
    # backward function
    def backward(self, X, y, output):
        # the derivative of the cross entropy loss applied to the softmax
        #   simplifies to predictions - labels
        error = output - y
        
        # relu
        if self.activation == 'relu':
            # Compute Hidden Layer Error: To propagate the error backward through the network, 
            #  we need to calculate the contribution of the hidden layer to the error. 
            #  This is done by taking the dot product of the error with the transposed weights 
            #   connecting the hidden layer to the output layer.
            hidden_error = np.dot(error, self.weights_hidden_output.T)
            hidden_delta = self.relu_derivative(self.hidden_output) * hidden_error 
            
        elif self.activation == 'sigmoid':
            hidden_error = np.dot(error, self.weights_hidden_output.T)
            hidden_delta = self.sigmoid_derivative(self.hidden_output) * hidden_error
        else:
            raise ValueError("Invalid activation function.")
        
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, error)
        self.bias_output -= self.learning_rate * np.sum(error, axis=0, keepdims=True)
        
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
        
    # cross-entropy loss
    def categorical_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        
    # train
    def train(self, X_train, y_train, epochs, dropout_p=0, decay_rate=1):
        for epoch in range(epochs):
            # shuffle training data
            # Shuffle indices
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            # Shuffle both arrays using the shuffled indices
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Forward pass
            output = self.forward(X_train_shuffled, is_training=True, dropout_p=dropout_p)
            
            # Backpropagation
            self.backward(X_train_shuffled, y_train_shuffled, output)
            
            # Compute and print loss
            loss = self.categorical_cross_entropy_loss(y_train_shuffled, output)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.4f}')
            
            # Update learning rate for the next epoch
            self.learning_rate = decay_rate * self.learning_rate  # Linear decay
                
    # predict
    def predict(self, X):
        return np.argmax(self.forward(X, is_training=False), axis=1)
