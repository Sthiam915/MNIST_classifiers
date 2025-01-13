import numpy as np
from scipy.special import softmax

from mnist import MNIST
import matplotlib.pyplot as plt
#I consider weights as originating from a node, and biases being at a node.


class Layer(object):
    def __init__(self, activation, weights, biases):
        self.activation = activation
        self.weights = weights
        self.biases = biases

#defining sigmoid function and its inverse for calculating 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(y):
    return np.log(y / (1 - y+2.220446049250313e-16))

# pretty self-explanatory. I currently have it explicitly 3 layers(784 - 15 - 10) and I want to 
# solve all the issues with the code before I generalize any of the algorithms to arbitrary layers
# and layer sizes
def feed_forward(layer1, layer2, layer3):
    layer2.activation =  layer1.weights @ layer1.activation + layer2.biases
    layer2.activation = sigmoid(layer2.activation)
        
    layer3.activation = layer2.weights @ layer2.activation + layer3.biases

    # layer3.activation = layer3.activation

    return (layer3.activation)

def backpropagation(layer1, layer2, layer3, ans):
    # I have split the gradient in 4: the gradient for weights in layer 1, gradient for weights in
    # layer 2, gradient for biases in layer 1,and gradient for biases in layer 2.
    gradientw2 = []
    gradientw1 = []
    
    gradientb2 = []
    gradientb1 = []
    
    step = 0.01    #Storing the partial derivatives I need when 
    accumulation = [0 for i in range(15)]
    
    # First, I calculate the gradients for the second layer. When I implemented this alone,
    # I managed to get decent accuracy(~50-60%), which indicated that I was on the right track
    for i in range(len(layer2.weights)):
        z = layer3.activation[i]
        y = int(i == ans)
        
        gradientb2.append(2 * (layer3.activation[i] - y))
        for j in range(len(layer2.weights[0])):
           
            gradientw2.append(2 * layer2.activation[j]  * (layer3.activation[i] - y))
            # Adding to the accumulation array to store the values I need, so I don't
            # have to go back and loop through for the summation
            
            accumulation[j] += 1 * (2 * layer2.weights[i][j]  *(layer3.activation[i] - y))
    
    # I thought my math  for the first layer was okay, but it seems sto have no effect on the network,
    # meaning when I isolate the gradient for the first layer, the results are pretty much random.
    for i in range(len(layer1.weights)):
        z1 = logit(layer2.activation[i])
        
        gradientb1.append(sigmoid(z1)*(1-sigmoid(z1)) * accumulation[i])
        for j in range(len(layer1.weights[0])):
            #print(i, j)
            #adding the accumulation to the gradient
            gradientw1.append(layer1.activation[i] * sigmoid(z1)*(1-sigmoid(z1)) * accumulation[i])
    
    #Applying the gradients to the weights and biases
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
    
 
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    #print(labels[:10])
    timages, tlabels = mndata.load_testing()
    #image = np.asarray(images[1000])
    input_layer = Layer(np.asarray(images[0])/255, (np.random.rand(15, 784)-0.5), [])
    
    #cv2.imwrite("display.jpg", image.reshape((28,28)))
    hidden_layer = Layer(np.zeros((1,15)), (np.random.rand(10, 15)-0.5)*2,(np.random.rand(15)-0.5)*2)
    output_layer = Layer(np.zeros((1,10)), [0], (np.random.rand(10)-0.5)*2)
    
    
    sum = 0

    feed_forward(input_layer, hidden_layer, output_layer)
    
    loss_tot = 0
    total = 20000
    x = []
    y_test = []
    y_train = []
    loss = []
    
    for i in range(total):
        
        input_layer.activation = np.asarray(images[i])/255
        feed_forward(input_layer, hidden_layer, output_layer)
        input_layer, hidden_layer, output_layer = backpropagation(input_layer, hidden_layer, output_layer, labels[i])
   
        

        loss_vec = np.zeros((10))
        loss_vec[labels[i]] = 1
        loss_tot +=  np.sum((output_layer.activation - loss_vec)**2)/20
        if i % 100 == 0:
            print(output_layer.activation)
            acc_test = test(timages, tlabels, input_layer, hidden_layer, output_layer)
            acc_train = test(images, labels, input_layer, hidden_layer, output_layer)
            x.append(i/total * 100)
            y_test.append(acc_test)
            y_train.append(acc_train)
            loss.append(loss_tot * 1.5)
            print("accuracy", acc_test, "%")
            print(f"trained on {i/total * 100}% of {total}")
            #print(f"Epoch {i}: Loss = {loss_tot/4000}")
            plt.plot(x,y_test, color='blue',label='test')
            plt.plot(x,y_train,color="red",label="train")
            plt.plot(x,loss,color="green",label="loss")
            plt.savefig("foo4.png")
            loss_tot = 0
    plt.show()

    
   
    
        
        
        
    
  
    
   
    

def display():
    pass
if __name__ == "__main__":
    main()