import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts


df = pd.read_csv('datasets/train.csv') # 42000x785
x = df.loc[:, 'pixel0':'pixel783'] # 42000X784
x = MinMaxScaler().fit_transform(x) # Normalizing Features
y = df['label'] # 42000x1

# Recodings labels as a matrix of 0s and 1s
y_mat = np.zeros((y.size, 10))
for i in range(y_mat.shape[0]):
    y_mat[i][y[i]] = 1

x_train, x_test, y_train, y_test = tts(x, y_mat, test_size=0.2, random_state=42)


class NeuralNetwork():
    def __init__(self, sizes, epochs=50, alpha=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.alpha = alpha
        
        # saving all parameters in this dictionary
        self.params = self.initialization()
        
    def sigmoid(self, z, derivative=False):
        exps = 1 / (1 + np.exp(-z))
        if derivative:
            return exps * (1 - exps)
        return exps
    
    def initialization(self):
        input_layer_size = self.sizes[0]
        hidden_1_size = self.sizes[1]
        hidden_2_size = self.sizes[2]
        output_layer_size = self.sizes[3]
        
        params = {
            'W1': np.random.randn(hidden_1_size, input_layer_size) * np.sqrt(1. / hidden_1_size),
            'W2': np.random.randn(hidden_2_size, hidden_1_size) * np.sqrt(1. / hidden_2_size),
            'W3': np.random.randn(output_layer_size, hidden_2_size) * np.sqrt(1. / output_layer_size)
        }
                
        return params
    
    def forward_propagate(self, x): 
        params = self.params
        
        # input layer becomes the sample
        params['A0'] = x
        
        # input layer to hidden layer 1
        params['Z1'] = np.dot(params['W1'], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        
        # hidden 1 to hidden layer 2
        params['Z2'] = np.dot(params['W2'], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
        
        # hidden layer 2 to the output layer
        params['Z3'] = np.dot(params['W3'], params['A2'])
        params['A3'] = self.sigmoid(params['Z3'])

        return params['A3']
    
    def back_propagate(self, y, pred):
        params = self.params
        gradients = {}
        
        delta3 = 2 * (pred - y) / pred.shape[0] * self.sigmoid(params['Z3'], derivative=True)
        gradients['grad3'] = np.outer(delta3, params['A2'])

        delta2 = np.dot(params['W3'].T, delta3) * self.sigmoid(params['Z2'], derivative=True)
        gradients['grad2'] = np.outer(delta2, params['A1'])
        
        delta1 = np.dot(params['W2'].T, delta2) * self.sigmoid(params['Z1'], derivative=True)
        gradients['grad1'] = np.outer(delta1, params['A0'])

        return gradients
    
    def compute_cost(self, x, y):
        m = x.shape[0]
        pred = self.forward_propagate(x.T)
        cost = (1/m) * sum(sum((-y * np.log(pred.T)) - ((1-y) * (np.log(1-pred.T)))))
        return cost
    
    def compute_accuracy(self, x_test, y_test):
        predictions = []
        
        for x,y in zip(x_test, y_test):
            pred = self.forward_propagate(x)
            prediction = np.argmax(pred)
            predictions.append(prediction == np.argmax(y))
            
        return np.mean(predictions) * 100
       
    def train(self, x_train, y_train, x_test, y_test):
        print('Training Neural Network...')
        params = self.params
        cur_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                pred = self.forward_propagate(x)
                gradients = self.back_propagate(y, pred)

                # updating parameters
                params['W1'] -= self.alpha * gradients['grad1']
                params['W2'] -= self.alpha * gradients['grad2']
                params['W3'] -= self.alpha * gradients['grad3']
                
            cost = self.compute_cost(x_train, y_train)
            
            accuracy = self.compute_accuracy(x_test, y_test)
            time_spent = round(time.time()-cur_time, 2)
            print(f'Epoch: {iteration+1} | Cost: {cost} | Time Spent: {time_spent}s | Accuracy: {accuracy}')
                
    
if __name__ == '__main__':
    nn = NeuralNetwork(sizes=[784, 128, 64, 10])
    nn.train(x_train, y_train, x_test, y_test)
    weights1 = np.array(nn.params['W1'])
    weights2 = np.array(nn.params['W2'])
    weights3 = np.array(nn.params['W3'])
    np.savetxt("weights1.txt", weights1)
    np.savetxt("weights2.txt", weights2)
    np.savetxt("weights3.txt", weights3)
    
