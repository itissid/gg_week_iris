import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load iris dataset
iris = datasets.load_iris()

# since this is a bunch, create a dataframe
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) 

# One-hot encode the 'class' column
encoder = preprocessing.OneHotEncoder()
iris_targets_one_hot = encoder.fit_transform(iris_df[['class']]).toarray()  # Convert to numpy array

# Extract the feature matrix X and the one-hot encoded labels Y
X = iris_df.drop('class', axis=1).values
Y = iris_targets_one_hot

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Define hyperparameters
lrn_rt = 0.001 #learning rate
epochs = 200
best_validation_loss = np.inf
patience_counter = 0
patience = 10
best_validation_loss = np.inf
validation_loss_history = []

#Variable History for MatPlotLib
validation_accuracy_history = []
training_loss_history = []

def reduce_learning_rate_if_plateaued(validation_loss_history, current_epoch, lr, factor=0.5, lr_patience=5, min_lr=1e-6):
    # Check if we have enough data to consider reducing learning rate
    if current_epoch > lr_patience:
        # Check if the loss has not improved in the last lr_patience epochs
        if min(validation_loss_history[-lr_patience:]) >= validation_loss_history[-(lr_patience + 1)]:
            # Reduce the learning rate if it's not already at the minimum
            new_lr = max(lr * factor, min_lr)
            return new_lr
    # Return the current learning rate if no reduction is necessary
    return lr

# define number of neurons in input, hidden and output layers 
n_input = 4
n_hidden = 10
n_output = 3

# Initialize weights and biases with He initialization for ReLU activation functions
np.random.seed(42)  # For reproducibility
he_std_dev_input_to_hidden = np.sqrt(2.0 / n_input)  # Standard deviation for He initialization
weights_input_to_hidden = np.random.randn(n_input, n_hidden) * he_std_dev_input_to_hidden

# Initialise weights and biases with Xavier initialisation for Softmax activation function 
xavier_std_dev_hidden_to_output = np.sqrt(2.0 / (n_hidden + n_output))
weights_hidden_to_output = np.random.randn(n_hidden, n_output) * xavier_std_dev_hidden_to_output

biases_hidden = np.zeros((1, n_hidden))  # Biases are initialized to zero
biases_output = np.zeros((1, n_output))  # Biases are initialized to zero

#Implement Rectified Linear Unit forward propagation step
def rectified(input_values):
   return np.maximum(0.0, input_values)

#Compute accuracy for thresholding
def compute_accuracy(predictions, labels):
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    return accuracy

#Forward propagation
def forward_propagation_to_hidden(input_matrix, weights, biases):
 hidden_layer_input = np.dot(input_matrix, weights) + biases
 hidden_layer_output = rectified(hidden_layer_input)
 return hidden_layer_output

# Define the softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward propagation from hidden to output layer
def forward_propagation_to_output(hidden_layer_output, weights, biases):
    # Calculate the input to the output layer
    output_layer_input = np.dot(hidden_layer_output, weights) + biases
    # Apply the softmax activation function
    output_layer_output = softmax(output_layer_input)
    return output_layer_output

#Compute the loss function
def categorical_cross_entropy(Y_true, Y_pred):
    # Clip the predictions to prevent log(0) error and floating point errors.
    Y_pred_clipped = np.clip(Y_pred, 1e-7, 1 - 1e-7)
    # Compute the cross-entropy loss
    loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / Y_true.shape[0]
    return loss

#Backpropagation step
def backward_propagation(X, Y, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output, hidden_layer_output, output_layer_output, learning_rate):
    # Compute the gradient of the loss with respect to the softmax logits (outer_layer_input) (softmax output - Y)
    dL_dout = output_layer_output - Y
    #(m,3) 

    # Compute the gradient of the loss with respect to the weights in the output layer
    dL_dW_output = np.dot(hidden_layer_output.T, dL_dout)
    #(m,3) * hidden layer output transpose (10, m) -> (10, 3)
    
    # Compute the gradient of the loss with respect to the biases in the output layer
    dL_db_output = np.sum(dL_dout, axis=0, keepdims=True)
    
    # Compute the derivative of the loss with respect to the hidden layer output
    dL_dhidden = np.dot(dL_dout, weights_hidden_to_output.T)
    #transposing the weights (3,10) * (m,3) -> (m,10)

    # Apply the derivative of the ReLU to the gradient
    dL_dhidden[hidden_layer_output <= 0] = 0

    # Compute the gradient with respect to the weights in the hidden layer
    dL_dW_hidden = np.dot(X.T, dL_dhidden)
    #transpose of input matrix (4,m) * (m, 10) -> (4, 10)
    
    # Compute the gradient with respect to the biases in the hidden layer
    dL_db_hidden = np.sum(dL_dhidden, axis=0, keepdims=True)

    # Update the weights and biases for the output layer
    weights_hidden_to_output -= learning_rate * dL_dW_output
    biases_output -= learning_rate * dL_db_output

    # Update the weights and biases for the hidden layer
    weights_input_to_hidden -= learning_rate * dL_dW_hidden
    biases_hidden -= learning_rate * dL_db_hidden

    return weights_input_to_hidden, biases_hidden, weights_hidden_to_output, biases_output

for epoch in range(epochs):
   # Shuffle the training data
    permutation = np.random.permutation(len(Y_train))
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train[permutation]

    #Shuffle validation data
    validation_permutation = np.random.permutation(len(Y_validation))
    X_validation_shuffled = X_validation[validation_permutation]
    Y_validation_shuffled = Y_validation[validation_permutation]

    # Forward propagation for the training dataset
    training_fp1 = forward_propagation_to_hidden(X_train_shuffled, weights_input_to_hidden, biases_hidden)
    training_fp2 = forward_propagation_to_output(training_fp1, weights_hidden_to_output, biases_output)

    # Forward propagation for the validation dataset
    validation_fp1 = forward_propagation_to_hidden(X_validation_shuffled, weights_input_to_hidden, biases_hidden)
    validation_fp2 = forward_propagation_to_output(validation_fp1, weights_hidden_to_output, biases_output)

    #Computing validation accuracy via thresholding
    validation_accuracy = compute_accuracy(validation_fp2, Y_validation_shuffled)
    validation_accuracy_history.append(validation_accuracy)

    #Compute loss function for the training data 
    train_loss = categorical_cross_entropy(Y_train_shuffled, training_fp2)
    training_loss_history.append(train_loss)

    #Compute loss function for the validation data
    validation_loss = categorical_cross_entropy(Y_validation_shuffled, validation_fp2)
    validation_loss_history.append(validation_loss) 

    print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {validation_accuracy:.4f}, Training Loss Function: {train_loss:.4f}, Validation Loss Function:{validation_loss:.4f}')

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        patience_counter = 0  # reset patience since we have new best validation loss
        # Saving the model parameters
        best_model_params = (weights_input_to_hidden.copy(), 
                             weights_hidden_to_output.copy(), 
                             biases_hidden.copy(), 
                             biases_output.copy())
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Early stopping: Validation loss did not improve for {} epochs".format(patience))
            break

    lrn_rt = reduce_learning_rate_if_plateaued(validation_loss_history, epoch, lrn_rt)

    # Backpropagation
    backward_propagation(X_train_shuffled, Y_train_shuffled, weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output, training_fp1, training_fp2, lrn_rt)

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Learning Rate: {lrn_rt}')


weights_input_to_hidden, weights_hidden_to_output, biases_hidden, biases_output = best_model_params

# Plotting validation accuracy over epochs
epoch_range = list(range(1, epochs + 1))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# First subplot for validation accuracy
ax1.plot(epoch_range, validation_accuracy_history, label='Validation Accuracy')
ax1.set_title('Validation Accuracy over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Validation Accuracy')
ax1.legend()

# Second subplot for training and validation loss
ax2.plot(epoch_range, training_loss_history, label='Training Loss')
ax2.plot(epoch_range, validation_loss_history, label='Validation Loss')
ax2.set_title('Training and Validation Loss over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

# Adjust the layout so that the plots do not overlap
plt.tight_layout()

# Display the figure
plt.show()

    


    

