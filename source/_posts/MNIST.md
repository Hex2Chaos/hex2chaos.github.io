---
title: The classic machine learning dataset—MNIST
categories:
  - AI
tags:
  - MNIST
date: 2024-12-25 23:00:00
---
<!--more-->
Today, we will locally test a classic machine learning dataset—MNIST (Modified National Institute of Standards and Technology database). This dataset is primarily used for training and testing image classification models, especially for handwritten digit recognition tasks.

## Introduction to the MNIST Dataset
### Content
- The MNIST dataset contains 70,000 grayscale images of handwritten digits.
- Each image is a 28x28 pixel grayscale image, with pixel values ranging from 0 to 255.
- The dataset is divided into two parts: 
  + Training set: 60,000 images.
  + Test set: 10,000 images.
- Each image corresponds to a label that indicates the digit (0 to 9) in the image.
### Applications of MNIST
- Image Classification: Training a model to recognize handwritten digits.  
- Model Validation: Testing the performance of machine learning models.  
- Algorithm Comparison: Comparing the effectiveness of different algorithms (such as SVM, KNN, neural networks, etc.).  
- Introduction to Deep Learning: Implementing classification tasks using simple neural networks (such as fully connected networks and convolutional neural networks).
### Characteristics of Models Trained on MNIST
- Simplicity
  Beginners can start with simple linear models (such as logistic regression or SVM).
  They can then try more complex models (such as multilayer perceptrons (MLP) or convolutional neural networks (CNN)).
- High Accuracy
  Due to the simplicity of the dataset, modern deep learning models (like CNNs) can easily achieve a test accuracy of 99%.
- Fast Training
  The dataset is small, allowing for quick model training, which is suitable for rapid iteration and experimentation.
- Scalability
  MNIST serves as the foundation for many more complex datasets. For example, Fashion-MNIST (for clothing classification) and EMNIST (Extended MNIST) are both variants based on MNIST

## Code
### import
```python
import tensorflow as tf
from tensorflow.keras import Sequential  
from tensorflow.keras.layers import Dense, Flatten  
from tensorflow.keras.datasets import mnist  
import matplotlib.pyplot as plt  
import numpy as np
```
> 1、import tensorflow as tf
Function: Import the TensorFlow library and abbreviate it as tf. TensorFlow is an open-source machine learning framework widely used for building and training deep learning models. Using tf allows for convenient access to various functionalities provided by TensorFlow, such as tensor operations, neural network construction, training, and more.
> 2、from tensorflow.keras import Sequential
Function: Import the Sequential class from TensorFlow's keras module.  
Sequential is a model-building method provided by Keras, suitable for stacking layers of a neural network in order.  
Using Sequential allows for the quick construction of simple neural network models.
> 3、from tensorflow.keras.layers import Dense, Flatten
Function: Import Dense and Flatten layers from Keras's layers module.  
Dense:  
A Fully Connected Layer, which is one of the most commonly used layers in neural networks. It connects the input to each neuron and calculates the output through weights and biases.  
Flatten:  
Flattens multi-dimensional input (such as a two-dimensional matrix of an image) into a one-dimensional vector. It is typically used to convert the output of convolutional layers into input for fully connected layers.
> 4、from tensorflow.keras.datasets import mnist
Function: Import the MNIST dataset from Keras's datasets module.  
MNIST is a classic handwritten digit dataset that contains images of handwritten digits from 0 to 9.  
Using mnist allows for easy loading of the training and testing sets, which can be directly used for model training and evaluation.
> 5、import matplotlib.pyplot as plt
Function: Import the pyplot module from Matplotlib and abbreviate it as plt.
Matplotlib is a library for data visualization, and pyplot is its submodule that provides MATLAB-like plotting capabilities.
Using plt allows you to create visualizations (such as displaying images of handwritten digits from the MNIST dataset).
> 6、import numpy as np
Function: Import the NumPy library and abbreviate it as np.  
NumPy is a library for scientific computing that provides efficient operations on multi-dimensional arrays.  
In deep learning, NumPy is commonly used for data processing (such as array operations, random number generation, matrix calculations, etc.).

### Load the MNIST Dataset
```python
# The MNIST dataset contains 60,000 training images and 10,000 testing images, with each image being a 28x28 grayscale image.  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
    
# Print the shape of the dataset to ensure data is loaded correctly  
print(f"Training set shape: {x_train.shape}, Label shape: {y_train.shape}")  
print(f"Test set shape: {x_test.shape}, Label shape: {y_test.shape}")
```
```text
output:
Training set shape: (60000, 28, 28), label shape: (60000,)  
Test set shape: (10000, 28, 28), label shape: (10000,)
```
> x_train: Contains the training set image data of the MNIST dataset. The data type is a NumPy array with a shape of (60000, 28, 28). It includes 60,000 images of handwritten digits, each image being a 28x28 grayscale image with pixel values ranging from 0 to 255.  
y_train: Contains the training set labels of the MNIST dataset. The data type is a NumPy array with a shape of (60000,). Each value is an integer representing the digit label (0 to 9) corresponding to the respective image.  
x_test: Contains the test set image data of the MNIST dataset. The data type is a NumPy array with a shape of (10000, 28, 28). It includes 10,000 images of handwritten digits, each image being a 28x28 grayscale image.  
y_test: Contains the test set labels of the MNIST dataset. The data type is a NumPy array with a shape of (10000,). Each value is an integer representing the digit label (0 to 9) corresponding to the respective test image.

### Data Visualization
```python
# Randomly view a few images to understand the data
def visualize_data(images, labels, num_samples=5):  
    # Define a function `visualize_data` for visualizing data.  
    # Parameters:  
    # - images: Image data (e.g., images from MNIST).  
    # - labels: Corresponding labels for the images.  
    # - num_samples: Number of samples to display, default is 5.  
    plt.figure(figsize=(10, 2))  
    # Create a new figure window, setting the image size to 10x2 inches.  
    for i in range(num_samples):  
        # Loop through the number of samples to display (from 0 to num_samples-1).  
        plt.subplot(1, num_samples, i + 1)  
        # Create a subplot with a layout of 1 row and num_samples columns, currently the i+1th subplot.  
        plt.imshow(images[i], cmap='gray')  
        # Display the ith image using a grayscale color map (cmap='gray').  
        plt.title(f"Label: {labels[i]}")  
        # Set the title of the subplot to show the corresponding label (e.g., "Label: 5").  
        plt.axis('off')
        # Turn off the axis display for the subplot.  
    plt.show()  
    # Display all subplots.
```
![MNIST images](/assets/2024-12-26-MNIST/visualize_data.jpg)

### Data Preprocessing
```python
# Normalization: Scale pixel values from 0-255 to 0-1 to enhance the efficiency and stability of model training.
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot Encoding of Labels: Convert labels from integers to one-hot encoded format, suitable for classification tasks.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### Building the Model
```
# Use Keras to construct a simple fully connected neural network (MLP) model.
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into one-dimensional vectors.
    Dense(128, activation='relu'),  # Hidden Layer 1: 128 neurons, activation function is ReLU.
    Dense(64, activation='relu'),   # Hidden Layer 2: 64 neurons, activation function is ReLU.
    Dense(10, activation='softmax') # Output Layer: 10 neurons (corresponding to digits 0-9), activation function is softmax.
])

# Print the model structure for easy inspection of the network's layers and parameter counts.
model.summary()
```
![model.jpg](/assets/2024-12-26-MNIST/model.jpg)
### Compile the Model
```python
# Specify the optimizer, loss function, and evaluation metrics
model.compile(
    optimizer='adam',  # Adam optimizer, suitable for beginners, automatically adjusts the learning rate
    loss='categorical_crossentropy',  # Categorical crossentropy loss function for multi-class classification
    metrics=['accuracy']  # Evaluation metric is accuracy
)
```
### Train the Model
```
# Train the model using the training data and validate its performance on the test set
history = model.fit(
    x_train, y_train,  # Training data
    validation_data=(x_test, y_test),  # Validation data
    epochs=10,  # Train for 10 epochs
    batch_size=32,  # 32 samples per batch
    verbose=2  # Display detailed information about the training process
)
```
### Evaluate the Model
```
# Evaluate the model's performance on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test set loss: {test_loss:.4f}")
print(f"Test set accuracy: {test_accuracy:.4f}")
```
### Visualize the Training Process
```
# Plot the accuracy curves for training and validation
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot the loss curves for training and validation
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
```
![accuracy_and_loss](/assets/2024-12-26-MNIST/accuracy_and_loss.jpg)
### using the Model for Prediction
```python
# Randomly select a test image for prediction
index = np.random.randint(0, x_test.shape[0])
test_image = x_test[index]
test_label = np.argmax(y_test[index])

# Prediction result
prediction = np.argmax(model.predict(test_image.reshape(1, 28, 28), verbose=0))

# Visualize the prediction result
plt.imshow(test_image, cmap='gray')
plt.title(f"Real label: {test_label}, Prediction: {prediction}")
plt.axis('off')
plt.show()
```
![benchmark](/assets/2024-12-26-MNIST/benchmark.jpg)
### Saving and Loading the Model
```
# Save the model to a file
model.save('mnist_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('mnist_model.h5')
print("The model has been saved and successfully loaded!")
```