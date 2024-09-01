import tensorflow as tf
import os
import cv2
import random
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

model_dir = "models"
abspath_model_dir = os.path.abspath(model_dir)
model = load_model(os.path.join(abspath_model_dir, "BinaryClassifier.keras"))

data_dir = "data"
abspath_happy_data_dir = os.path.join(os.path.abspath(data_dir), "Happy")
happy_image_files = [file for file in os.listdir(abspath_happy_data_dir) if os.path.isfile(os.path.join(abspath_happy_data_dir, file))]
abspath_sad_data_dir = os.path.join(os.path.abspath(data_dir), "Sad")
sad_image_files = [file for file in os.listdir(abspath_sad_data_dir) if os.path.isfile(os.path.join(abspath_sad_data_dir, file))]
image_files = happy_image_files + sad_image_files

image_chosen = random.choice(image_files)

if image_chosen in happy_image_files:
    abspath_image_chosen = os.path.join(abspath_happy_data_dir, image_chosen)
    state = "Happy"
else:
    abspath_image_chosen = os.path.join(abspath_sad_data_dir, image_chosen)
    state = "Sad"

image = cv2.imread(abspath_image_chosen)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tf.image.resize(image, (256,256))

predicted_state_value = model.predict(np.expand_dims(image/255,0))

if predicted_state_value > 0.5:
    predicted_state = "Sad"
else:
    predicted_state = "Happy"

print("Actual State: ", state)
print("Predicted State: ", predicted_state)

plt.imshow(image.numpy().astype(int))
plt.show()