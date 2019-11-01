import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

# ----- Prepare Data ----- #
# preparing your data including data normalization
batch_size = 128
num_classes = 10
epochs = 50

img_rows, img_cols = 28, 28

x_train = (train_data.iloc[:, 1:].values).astype('float32')
y_train = (train_data.iloc[:, 0].values).astype('int32')
x_test = (test_data.iloc[:, 1:].values).astype('float32')
y_test = (test_data.iloc[:, 0].values).astype('int32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (img_rows, img_cols, 1)

# ----- Build CNN Network ----- #
# Define your model here
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ----- Define your loss function, optimizer and metrics ----- #
model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

# ----- Complete PlotLearningCurve function ----- #
def PlotLearningCurve(epoch, trainingloss, testingloss):
    plt.plot(epoch, trainingloss, 'b', label='training loss')
    plt.plot(epoch, testingloss, 'b', label='testing loss', color='red')
    plt.title('Learning curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('lc.png')

# fit your model by using training data and get predict label
history = model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(x_test, y_test))

# plot learning curve
# PlotLearningCurve(pass)

# evaluation your model by using testing data
score = model.evaluate(x_test, y_test, verbose=0)
print("final test accuracy", score)
PlotLearningCurve(range(len(history.history['loss'])), history.history['loss'], history.history['val_loss'])
