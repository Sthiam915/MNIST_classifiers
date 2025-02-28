from mnist import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#classify mnist digits: 784 -> 15 -> 10
class Classify_Digits(nn.Module):
    def __init__(self):
        super(Classify_Digits, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc_1 = nn.Linear(32 * 7 * 7, 25)
        self.fc_2 = nn.Linear(25, 10)
    def forward(self, x):
        #sigmoid initially because that's what I was using for my custom one,, change to reLU later
        x = F.relu(self.conv_1(x))
        x = self.pool(x)
        x = F.relu(self.conv_2(x))
        x = self.pool(x)

        x = x.view(1, 1568)
       
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

def test(images, labels, model, width):
    correct = 0
    for i in range(width):
        guess = model(torch.tensor(images[i], dtype=torch.float32).view(1, 28, 28)/255) 
        #print(f"guess:{guess}\nanswer:{labels[i]} | {torch.argmax(guess) }")
        if torch.argmax(guess) == labels[i]:
            correct += 1
    return correct/width * 100

def main():
    #get inputs and testing data
    mnist_data = MNIST("samples")
    training_images, training_labels = mnist_data.load_training()
    testing_images, testing_labels = mnist_data.load_testing()
    

    #initialize model
    digit_classifier = Classify_Digits()
    #using MSE because my practice NN was using MSE, will change to cross-entropy later
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(digit_classifier.parameters(), lr = 0.0005)
    x = []
    y2 = []
    y = []
    saved_accuracy = open("best_cnn.txt",'r')
    best_accuracy = float(saved_accuracy.read())
    saved_accuracy.close()
    for k in range(20):
        for i in range(50000):
            optimizer.zero_grad()
            
            out = digit_classifier(torch.tensor(training_images[i], dtype=torch.float32).view(1, 28, 28)/255)
            label = torch.tensor(training_labels[i], dtype=torch.long).view(-1)
            loss = cost_function(out, label)
            loss.backward()
            optimizer.step()
            if i > 20000:
                optimizer = optim.SGD(digit_classifier.parameters(), lr = 0.001)
            if i > 25000:
                optimizer = optim.SGD(digit_classifier.parameters(), lr = 0.00005)
            if i > 30000:
                optimizer = optim.SGD(digit_classifier.parameters(), lr = 0.001+0.005/(k+1))
            if i > 45000:
                optimizer = optim.SGD(digit_classifier.parameters(), lr = 0.00002)
            if (i+1) % 100 == 0 :
                if not k > 2:
                    print(f"epoch: {k} , Iteration {i+1 + 50000 * k}, Loss: {loss.item():.4f}")
                    continue
                pct1 = test(testing_images, testing_labels, digit_classifier, 10000)
                pct2 = test(training_images, training_labels, digit_classifier,1)
                x.append(i)
                y.append(pct1)
                y2.append(pct2)
                plt.plot(x,y, color="red")
                #plt.plot(x,y2, color="blue")
                plt.savefig('foo6.png')
                print(f"epoch: {k} , Iteration {i+1 + 50000 * k}, Loss: {loss.item():.4f}, Testing Accuracy: {pct1} %, Training Accuracy: {pct2} %")
                if(pct1 > best_accuracy and k> 2):
                    saved_accuracy = open("best_cnn.txt",'+w')
                    saved_accuracy.write(str(pct1))
                    best_accuracy = pct1
                    torch.save(digit_classifier.state_dict(), "cnn_params.pth")
                    saved_accuracy.close()
    
    
    
        
if __name__ == "__main__":
    main()