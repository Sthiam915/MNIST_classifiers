import torch
from mnist import MNIST
import matplotlib.pyplot as plt
import nn_torch
import cnn
import cv2
import numpy
test_images, test_labels = MNIST("samples").load_testing()

cnn_classifier = cnn.Classify_Digits()
ff_classifier = nn_torch.Classify_Digits()

cnn_classifier.load_state_dict(torch.load("cnn_params.pth"))
ff_classifier.load_state_dict(torch.load("nn_params.pth"))

# cnn_accuracy = cnn.test(test_images, test_labels, cnn_classifier, 10000)
# ff_accuracy = nn_torch.test(test_images, test_labels, ff_classifier, 10000)
for i in range(10000):
    guess = cnn_classifier(torch.tensor(test_images[i], dtype=torch.float32).view(1, 28, 28)/255) 
    if not torch.argmax(guess) == test_labels[i]:
            print(test_labels[i])
            
            win_name = f"calculated:{torch.argmax(guess)}, true:{test_labels[i]}"
            print(numpy.asarray(test_images[i]).reshape((28,28)))
            img = numpy.asarray(test_images[i]).reshape((28,28))
            img = img.astype('uint8')
            cv2.imshow(winname=win_name, mat=img)

