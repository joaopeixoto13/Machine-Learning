import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
fashion_mnist = keras.datasets.fashion_mnist

# Split data into Train and Test
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 60000 images that are made up of 28x28 pixels
print(train_images.shape)
print(type(train_images))

# Have look one pixel (value 0-255 gray scale)
print(train_images[0,23,23])

# let's have a look at the first 10 training labels
print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])     # Show first image
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Adjust the scale (0-1) --> better results
train_images = train_images / 255.0
test_images = test_images / 255.0


# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # input layer (1)
    keras.layers.Dense(128, activation='relu'),     # hidden layer (2)
    keras.layers.Dense(10, activation='softmax')    # output layer (3)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
model.fit(train_images, train_labels, epochs=10)  # We pass the data, labels and epochs and watch the magic!


# Evaluating the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('Test accuracy:', round(test_acc * 100, 2), '%')


# Making Predictions
# predictions = model.predict(test_images)
# print(predictions[0])                                # print the list of predictions for element 0
#
# print(class_names[np.argmax(predictions[0])])         # Find the max probability (find the class)
# plt.figure()
# plt.imshow(test_images[0])                            # Show first image
# plt.colorbar()
# plt.grid(False)
# plt.show()


# Verifying Predictions
# I've written a small function here to help us verify predictions with some simple visuals.

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)