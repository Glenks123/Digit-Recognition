import numpy as np
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


# df = pd.read_csv('datasets/train.csv') # 42000x785
# x = df.loc[:, 'pixel0':'pixel783'] # 42000X784
# x = MinMaxScaler().fit_transform(x) # Normalizing Features
# sample = 1
# image = x[sample]
# image = image.reshape(28, 28)
# fig = plt.figure
# plt.imshow(image)
# plt.show()

# Loading Optimized Weights
W1 = np.loadtxt('weights1.txt')
W2 = np.loadtxt('weights2.txt')
W3 = np.loadtxt('weights3.txt')


def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def predict_digit(img):
    # Resizing the image to 28x28 pixels
    img = img.resize((28, 28))
    # Converting from RGB to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(-1, 1)
    # Normalizing img
    img = MinMaxScaler().fit_transform(img)
    # predicting by feed forward propagating
    # for i in range(img.shape[0]):
    #     if (img[i] == 1.):
    #         img[i] = 0
    # print(img)
    print(img)
    A1 = sigmoid(np.dot(W1, img))
    A2 = sigmoid(np.dot(W2, A1))
    A3 = sigmoid(np.dot(W3, A2))
    prediction = A3
    return np.argmax(prediction)
    
    
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "black", cursor="cross")
        self.label = tk.Label(self, text="Welcome", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting) 
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2)
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit = predict_digit(im)
        self.label.configure(text=str(digit))
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=10
        self.canvas.create_rectangle(self.x-r, self.y-r, self.x + r, self.y + r, fill='white')
        
        
if __name__ == '__main__':
    app = App()
    app.mainloop()