import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = "data"
abspath_data_dir = os.path.abspath(data_dir)

#Returns a tf.data.Dataset from the directory.
data = tf.keras.utils.image_dataset_from_directory(abspath_data_dir)

#The tf.data.Dataset, or our "data", is in the form of tf.tensors. In order to look at the values inside them we need to make them numpy arrays.
#The data_iterator is a generator that generates a batch of numpy arrays as its output, when it is called.
data_iterator = data.as_numpy_iterator()
#The batch is a tuple of the form (Image, Label). Both have 32 sets as that is the size of a single batch created in line 12.
batch = data_iterator.next()

#View a batch of 32 images and their labels. 0 = Happy, 1 = Sad.
figures, axis = plt.subplots(nrows=8, ncols=4, figsize=(20,20))
axis = axis.flatten() 
for id, img in enumerate(batch[0][:32]):
    axis[id].imshow(img.astype(int))
    axis[id].title.set_text(batch[1][id])
    axis[id].axis('off')
plt.tight_layout()
plt.show()