# MNIST Classifiers

This projects contains various Neural Network-based classifiers for the MNIST handwritten dataset. I have a special focus on the neural network built from scratch

## Installation

To install all dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To use the from scratch, simply run the command:

```
python3 nn_from_scratch.py [train_size] [learning_rate]
```

```
# example usage
python3 nn_from_scratch.py 30000 0.01
```

This example will train the network on 30,000 images with a learning rate of 0.01. train_size can be from 0 to 50000, and learning_rate must be greater than 0.

The network will continuously output its current test accuracy as well as how much of the set it has been trained on.

When the network is finished training, it will output a png image "accuracy.png" containing a graph displaying training accuracy, testing accuracy, and loss.

# example outputs

![alt text](examples/terminal.png)
![alt text](examples/accuracy.png)


## Future Implementations

* Implement Network class for flexible network architecture
* Save weights and biases so we don't have to train network every time
* Implement better training schemes(variable learning rate, options for multiple epochs etc)
* Create GUI so users can try to classify their own handwriting
