import numpy as np
from scipy.special import softmax
import sys
from mnist import MNIST
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, activation, weights, biases):
        self.activation = activation
        self.weights = weights
        self.biases = biases


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def feed_forward(layer1, layer2, layer3):
    layer2.activation =  layer1.weights @ layer1.activation + layer2.biases
    layer2.activation = sigmoid(layer2.activation)
    layer3.activation = layer2.weights @ layer2.activation + layer3.biases

    return (layer3.activation)

# backpropagates by calculating gradient based on whether ans is correct
# plan on adding step parameter instead of hardcoding
def backpropagation(layer1, layer2, layer3, ans, step):

    # store gradients for each layer
    gradientw2 = []
    gradientw1 = []
    
    gradientb2 = []
    gradientb1 = []
    
       
    
    # store gradients for previous layers
    accumulation = [0 for i in range(15)]
    
    for i in range(len(layer2.weights)):
        z = layer3.activation[i]
        y = int(i == ans)
        
        
        gradientb2.append(2 * (layer3.activation[i] - y))
        for j in range(len(layer2.weights[0])):
            
            gradientw2.append(2 * layer2.activation[j]  * (layer3.activation[i] - y))            
            accumulation[j] += 1 * (2 * layer2.weights[i][j]  *(layer3.activation[i] - y))

    for i in range(len(layer1.weights)):
        z1 = layer2.activation[i]
        
        gradientb1.append(z1*(1-z1) * accumulation[i])
        for j in range(len(layer1.weights[0])):
            
            gradientw1.append(layer1.activation[j] * z1*(1-z1) * accumulation[i])
    
    # applying the gradients to the weights and biases
    layer2.weights = np.reshape(np.ravel(layer2.weights) -  (step * np.array(gradientw2)), (10, 15) )
    layer3.biases = layer3.biases - (step *  np.array(gradientb2))
    layer1.weights = np.reshape(np.ravel(layer1.weights) - (step * np.array(gradientw1)), (15, 784) )
    layer2.biases = layer2.biases - (step *  np.array(gradientb1))
    
    return (layer1, layer2, layer3)


def test(timages,tlabels, input_layer, hidden_layer, output_layer):
    sum = 0
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(100):
        
        input_layer.activation = np.asarray(timages[i])/255
        
        guess = np.argmax(softmax(feed_forward(input_layer, hidden_layer, output_layer)))
        
        
        dist[tlabels[i]] +=1
        if(guess  == tlabels[i]):
            nums[tlabels[i]] += 1
            sum += 1
    for i in range(len(nums)):
        nums[i] = nums[i]/dist[i]

    return sum/100 * 100
    
def main():
    step = float(sys.argv[2])
    train_size = int(sys.argv[1])
    if not train_size >= 0 and train_size < 50001:
        raise Exception("Training size must be between 0 and 50000")
 
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    timages, tlabels = mndata.load_testing()

    # currently, initialization is hardcoded -- I will add robust initialization to layer class
    # and maybe make a network class
    input_layer = Layer(np.asarray(images[0])/255, (np.random.rand(15, 784)-0.5), [])
    hidden_layer = Layer(np.zeros((1,15)), (np.random.rand(10, 15)-0.5)*2,(np.random.rand(15)-0.5)*2)
    output_layer = Layer(np.zeros((1,10)), [0], (np.random.rand(10)-0.5)*2)
    
    feed_forward(input_layer, hidden_layer, output_layer)
    
    loss_tot = 0
  
    x = []
    y_test = []
    y_train = []
    loss = []
    
    for i in range(train_size):
        
        input_layer.activation = np.asarray(images[i])/255
        feed_forward(input_layer, hidden_layer, output_layer)
        input_layer, hidden_layer, output_layer = backpropagation(input_layer, hidden_layer, output_layer, labels[i], step)
   
        

        loss_vec = np.zeros((10))
        loss_vec[labels[i]] = 1
        loss_tot +=  np.sum((output_layer.activation - loss_vec)**2)/20
        if i % 100 == 0:
           
            acc_test = test(timages, tlabels, input_layer, hidden_layer, output_layer)
            acc_train = test(images, labels, input_layer, hidden_layer, output_layer)
            x.append(i/train_size * 100)
            y_test.append(acc_test)
            y_train.append(acc_train)
            loss.append(loss_tot * 1.5)
            print("accuracy", acc_test, "%")
            print(f"trained on {i/train_size * 100}% of {train_size} images")
            print("---------------------------")
        
            
            
            loss_tot = 0
    plt.plot(x,y_test, color='blue',label='test')
    plt.plot(x,y_train,color="red",label="train")
    plt.plot(x,loss,color="green",label="loss")
    plt.ylabel("percentage accuracy")
    plt.xlabel("% of images trained")
    plt.legend()
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    main()