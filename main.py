import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data():
    mat1 = loadmat('data.mat')
    X = mat1['X'] # (5000, 400) dimensional matrix. 400 = number of features (20x20 pixel image). 5000 = no. of training examples
    y = mat1['y'] # (5000, 1) dimensional vector
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def rand_initialize_weights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, L_in + 1) * (2*epsilon_init) - epsilon_init
    return W

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):
    # Reshape n_params to get Theta1 and Theta1
    Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

    # Forward Propagating
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X)) # adding bias units to input layer
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2))) # adding the bias term to hidden layer
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    pred = a3
    
    # Recode the labels as a matrix of 0s and 1s where each row has an one element 1 representing the actual value
    y_vec = np.zeros((m, num_labels))
    for i in range(0, y_vec.shape[0]):
        y_vec[i][y[i]-1] = 1
    
    costJ = (1/m) * sum(sum((-y_vec * np.log(pred)) - ((1-y_vec) * (np.log(1-pred)))))
    # return costJ
    
    # Implementing backpropagation to compute the partial derative terms (gradients)
    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))
    
    for i in range(m):
        X_t = X[i, :] # x_t is a vector containing the grayscale value for each training set. 1 x 401 dimension
        y_t = y_vec[i, :]
        
        a1_t = X_t
        z2_t = np.dot(a1_t, Theta1.T)
        a2_t = np.hstack((1, sigmoid(z2_t)))
        z3_t = np.dot(a2_t, Theta2.T)
        a3_t = sigmoid(z3_t) # hypotheses prediction

        # Backpropagating
        delta3 = a3_t-y_t
        del_2 = np.dot(Theta2.T, delta3.T)
        delta2 = (del_2[1:del_2.shape[0]]).T * sigmoid_gradient(z2_t)

        # Accumulating the partial derivative terms
        # delta2'*a1 computes the partial derivative terms (gradient)
        # print(np.array([delta2]).T.shape, np.array([X_t]).shape)
        grad1 = grad1 + np.dot(np.array([delta2]).T, np.array([X_t]))
        grad2 = grad2 + np.dot(np.array([delta3]).T, np.array([a2_t]))
        
    grad1 = grad1 * (1/m)
    grad2 = grad2 * (1/m)
    
    return costJ, grad1, grad2

def gradient_descent(X, y, initial_nn_params, alpha, iterations, input_layer_size, hidden_layer_size, num_labels):
    # Reshape initial_nn_params to get Theta1 and Theta1
    Theta1 = initial_nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = initial_nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    
    m = len(y)
    J_history = []
    
    for i in range(iterations+1):
        nn_params = np.append(Theta1.flatten(), Theta2.flatten())
        J, grad1, grad2 = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        J_history.append(J)
        print(f'Iteration: {i} | Cost: {J}')
        
    nn_params_final = np.append(Theta1.flatten(), Theta2.flatten())
    return nn_params_final, J_history

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X)) # adding bias units to input layer
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2))) # adding the bias term to hidden layer
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    pred = a3 
    
    return np.argmax(pred,axis=1)+1
    
def plot_J_against_iters(J_history):
    plt.plot(J_history)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost function J')
    plt.show()
            
X, y, Theta1, Theta2 = load_data()

INPUT_LAYER_SIZE = 400 # 20x20 pixel image
HIDDEN_LAYER_SIZE = 25 # 25 hidden units
NUM_LABELS = 10 # 10 labels. 1-10
# nn_params = np.append(Theta1.flatten(), Theta2.flatten()) # Unrolling all parameters into a 10285 x 1 dimensional vector
# J, grad1, grad2 = nn_cost_function(nn_params, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, NUM_LABELS, X, y)
# print(f'Cost: {J}')

initial_Theta1 = rand_initialize_weights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) # (25, 401) dimensions
initial_Theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS) # (10, 26) dimensions
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten()) # (10285,) dimension

print('Training neural network...')
nn_theta, J_history = gradient_descent(X, y, initial_nn_params, 0.8, 800, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, NUM_LABELS)
print(f'Minimized Cost: {J_history[-1]}')

optimized_Theta1 = nn_theta[:((INPUT_LAYER_SIZE+1) * HIDDEN_LAYER_SIZE)].reshape(HIDDEN_LAYER_SIZE,INPUT_LAYER_SIZE+1)
optimized_Theta2 = nn_theta[((INPUT_LAYER_SIZE +1)* HIDDEN_LAYER_SIZE ):].reshape(NUM_LABELS,HIDDEN_LAYER_SIZE+1)

prediction = predict(optimized_Theta1, optimized_Theta2, X)
print("Training Set Accuracy:",sum(prediction[:,np.newaxis]==y)[0]/5000*100,"%")

plot_J_against_iters(J_history)