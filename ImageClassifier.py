# Made By James Jasper Faddeno O'Roarke.
# Credits to Datacamp for giving us guided tutorials on how to create neural networks, teaching us how neural networks work in-depth, and giving us an online API to test our code in without setting up an environment. Credits to Samuel Brockett for some extra help in a few places. Additionally, Credits to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for creating the Cifar10 dataset for us to use and train our network with.

# This program creates a neural network that predicts what inputted images are out of 10 broad categories.

# Before any programming, some modules must be imported to Python.
import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential

# Initialize our classifier model as a sequential model, one that goes through it's layer from start to finish.
model = Sequential()

# Before creating any parts of our model, we have to load the dataset.
# During this, we also divide our dataset into 4 variables: two for training the classifier and two for testing the classifier. The reason it is important to set aside data for testing is that neural networks can "overfit," or find patterns in the training data that don't exist in real life. Setting aside test data however, gives us an accurate idea of how well the neural network was trained so that we can optimize it.
# First, we create two dataImage and dataLabel variables and load cifar10 into them. The trainData variables contain 50000 images/labels, and the testData variables contain 10000 images/labels.
(trainDataImage, trainDataLabel), (testDataImage, testDataLabel) = cifar10.load_data()


# Create our first layer. This will be a 2D Convolution layer, a special kind of layer that is powerful for processing images. It allows us to create a specific number of specifically sized "kernels," which are special matrices (or small images) of pixels that the model uses to find patterns in images, like a 3x3 horizontal line that is used to see the bottom and top of an object in an image. Additionally, it functions as an input layer when we specify the model shape (width, height, and depth/color).
# In this case, we create a layer with 15 3x3 kernels. We also use an activation function (which is a mathematical operation applied to every output of a layer's node) of ReLU, a simple activation function that makes all negative outputs zero to introduce non-linearity to the output. For our input shape, we give 32 for width, 32 for height, and 3 for depth because we have colored 32x32 images. 
# Additionally, we add padding. This gives the image a white, 1-pixel border that prevents data loss from using the kernel. The kernel will reduce the resolution of the image by one per kernel size over 1, and because we are using a 3x3 kernel and using 'same' padding (which makes the image have a 2 point larger width and height), the output is the same as the input.
model.add(Conv2D(80, kernel_size = 3, activation='relu', input_shape=(32, 32, 3), padding='same'))

# Here we add a "dropout" layer. This layer deactivates a percentage (in this case 20%) of the previous layer's node randomly during each epoch. This leads to slower training but helps prevent overfitting.
model.add(Dropout(0.2))

# Here we add another convolution layer, although this time the model does not have to have the input_shape command because it has already received its inputs from this point. 
model.add(Conv2D(60, kernel_size = 3, activation='relu', padding='same'))

model.add(Dropout(0.2))

# Here we add a pooling layer. This decreases the resolution of the image from here in order to make the model have less trainable parameters. This causes the model to train more quickly and more easily and allows the neural network to be deeper than it otherwise could be.
model.add(MaxPool2D(2))

# We add another Conv2D layer.
model.add(Conv2D(50, kernel_size = 3, activation='relu', padding='same'))

model.add(Dropout(0.2))

model.add(MaxPool2D(2))

# We add our final Conv2D layer. Each new layer has its kernel size times more trainable parameters, so it is useful to narrow down the number of kernels over layers.
model.add(Conv2D(40, kernel_size = 3, activation='relu', padding='same'))

model.add(Dropout(0.2))

# Now, we add a flatten layer. This causes the previous layer's outputs, which are in a group of matrices/images, to be converted into a massive list of pixels, allowing it to be processed into the output
model.add(Flatten())

# Here, we create the output layer. Due to having ten possible outputs from 10 possible labels for the given image, the dense layer must have 10 possible nodes. Additionally, we use the softmax function, which turns given values into probabilities between 0 and 1. This is very powerful for outputs of classification models, as it gives a percent possibility of each label corresponding to the image, and then lets the most probable label being used later on.
model.add(Dense(10, activation='softmax'))

# Creates an output to the shell showing the structure and number of parameters of the model. This helps with debugging and enhancing the classifier.
model.summary()

# Now that we have finished the structure of our classifier, we must begin to process our data. First, we define a function to simplify the reshaping of our data
# This function will take a variable, number of rows (corresponds to the number of images/labels in the variable), and check for a boolean that shows whether the variable being reshaped is an image or label
def Reshaper(var, imNumber, isImage):
    if(isImage==True):
        np.reshape(var, (imNumber, 32, 32, 3))
    else:
        np.reshape(var, (imNumber, 1))
    return var

# Now we can simply call our variable with 50000 images
trainDataImage = Reshaper(trainDataImage, 50000, True)
trainDataLabel = Reshaper(trainDataLabel, 50000, False)

# One step we have to take before training the model is to turn our data into "one-hot encoded arrays." Essentially, we turn the DataLabels into 10 (due to having 10 possible options) wide matrixes of 0s and 1s. In it, the value that corresponds to the category that the row belongs to is 1, but all other values are zero.
# In order to do this, we will have to define a special function. This will take a DataLabel array and change it into a one-hot encoded array.

def OneHotEncode(DataLabel, labelNum):
    # We create a labelNum by 10 matrix of zeros that will be our model
    OneHot = np.zeros((labelNum, 10))

    # We also initialize a counter variable at zero
    count=0

    # Now we go to every row of the matrix and set the column that the row corresponds with to one
    while (count < labelNum):
        OneHot[count][DataLabel[count]] = 1

        count=count+1

    return OneHot

# Now we call our function to encode trainDataLabel
trainDataLabel = OneHotEncode(trainDataLabel, 50000)


# Here, we compile our data, readying it for training. We assign an optimizer that controls various aspects of the model's training and assign 'adam' here due to being a powerful general-purpose optimizer
# We assign a loss function, which is a mathematical function that is used to guide the model on how to improve by showing how inaccurate it is in various ways. We use categorical cross-entropy, as it a useful function for classifier models
# Finally, we add a metric, which gives feedback to the user on how well the model is doing. We add accuracy on the metric to get a percentage on how accurate the model is at estimating what an image is
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Now, we do the longest and most important step of training the model.
# Here, we give the model a training dataset to estimate from and a training answer set to score itself on, which correspond to the images and labels respectively.
# We also specify the number of epochs, which controls how long the model trains for (how many times it fully goes through the data), and the batch size, which controls how many images the model trains on at a time. It is important to be careful of the number of epochs, as a small number leads to an untrained model and a large number leads to an overfit model (one only specialized to the training set)
model.fit(trainDataImage, trainDataLabel, epochs=40, batch_size=100)


# Now we call all of the functions we previously made to format the testing data
testDataImage = Reshaper(testDataImage, 10000, True)
testDataLabel = Reshaper(testDataLabel, 10000, False)
testDataLabel = OneHotEncode(testDataLabel, 10000)

# Now that our model is trained, we must evaluate it against testing data it has never seen before and produce the program's score

output = model.evaluate(testDataImage, testDataLabel, verbose=True, batch_size=100)

# Finally, output the model's loss to the user and save it as a file.
print("The model loss and accuracy respectively:", output)
model.save('mlmodel.h5')









