import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = "data"
abspath_data_dir = os.path.abspath(data_dir)

data = tf.keras.utils.image_dataset_from_directory(abspath_data_dir)

#'data.map' directly allows us to map a function to the tf.data.Dataset
data = data.map(lambda x, y : (x/255, y))

#In our case there are 13 batches in data (identified using print(len(data))).
train_size = int(len(data)*.7)  #9 batches are used for training.
val_size = int(len(data)*.2)    #2 batches are used for validation.
test_size = int(len(data)*.1)+1 #2 batches are used for testing.
#When added, we get back our original data batch number.

#'data.take' allows us to grab that number of batches, and 'data.skip' skips over the specified number of batches.
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Building the CNN Model
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), 1, activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

#Compliling the model along with the optimizer of choice, the loss function, and the metrics to track. 
model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Setting up the logging of the data returned from trianing.
log_dir = "logs"
abspath_log_dir = os.path.abspath(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=abspath_log_dir)

#Training the model.
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

#Plotting the training metrics to evaluate the training of the model.
fig = plt.figure()
plt.plot(hist.history["loss"], color="teal", label="loss")
plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
fig.suptitle("Loss")
plt.legend(loc="upper right")

fig2 = plt.figure()
plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
fig2.suptitle("Accuracy")
plt.legend(loc="upper right")

plt.show()

#Save the model.
model_dir = "models"
abspath_model_dir = os.path.abspath(model_dir)
model.save(os.path.join(abspath_model_dir, "BinaryClassifier.keras"))