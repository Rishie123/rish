#!/usr/bin/env python
# coding: utf-8

# # Satellite Image Classification

# Step 1 : Install all the packages needed in cmd :-  tensorflow, tensorflow_addons, tensorflow_datasets, tensorflow_hub, numpy, matplotlib and sckitlearn using:
# ### py -m pip install [packagename]

# Step 2: Install all the libraries needed from those packeges

# In[ ]:


import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa


# Step 3: Import the Datasets

# In[ ]:


# load the whole dataset, for data info

all_ds   = tfds.load("eurosat", with_info=True)

# load training, testing & validation sets, splitting by 60%, 20% and 20% respectively

train_ds = tfds.load("eurosat", split="train[:60%]")
test_ds  = tfds.load("eurosat", split="train[60%:80%]")
valid_ds = tfds.load("eurosat", split="train[80%:]")


# Step 4: Create variables for storing classes

# In[ ]:


# the class names
class_names = all_ds[1].features["label"].names
# total number of classes (10)
num_classes = len(class_names)
num_examples = all_ds[1].splits["train"].num_examples


# Step 5: Understanding the Data by plotting

# In[ ]:


# make a plot for number of samples on each class

fig, ax = plt.subplots(1, 1, figsize=(14,10))
labels, counts = np.unique(np.fromiter(all_ds[0]["train"].map(lambda x: x["label"]), np.int32), 
                       return_counts=True)

plt.ylabel('Counts')
plt.xlabel('Labels')
sns.barplot(x = [class_names[l] for l in labels], y = counts, ax=ax) 
for i, x_ in enumerate(labels):
  ax.text(x_-0.2, counts[i]+5, counts[i])
# set the title
ax.set_title("Bar Plot showing Number of Samples on Each Class")
# save the image
# plt.savefig("class_samples.png")


# Step 6: Preparing the data for training

# In[ ]:


def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
  # shuffle the dataset
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Repeat forever
  ds = ds.repeat()
  # split to batches
  ds = ds.batch(batch_size)
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds


# cache(): This method saves the preprocessed dataset into a local cache file. This will only preprocess it the very first time (in the first epoch during training).
# map(): We map our dataset so each sample will be a tuple of an image and its corresponding label one-hot encoded with tf.one_hot().
# shuffle(): To shuffle the dataset so the samples are in random order.
# repeat()Every time we iterate over the dataset, it'll repeatedly generate samples for us; this will help us during the training.
# batch(): We batch our dataset into 64 or 32 samples per training step.
# prefetch(): This will enable us to fetch batches in the background while the model is training.

# Step 7: Run the model for training and validation

# In[ ]:


# validating shapes
for el in valid_ds.take(1):
  print(el[0].shape, el[1].shape)
for el in train_ds.take(1):
  print(el[0].shape, el[1].shape)


# Output:

# In[ ]:


(64, 64, 64, 3) (64, 10)
(64, 64, 64, 3) (64, 10)


# Fantastic, both the training and validation have the same shape; where the batch size is 64, and the image shape is (64, 64, 3). The targets have the shape of (64, 10) as it's 64 samples with 10 classes one-hot encoded.

# Step 8: Building The Model

# model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"
# 
# download & load the layer as a feature vector
# keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)
# 
# m = tf.keras.Sequential([
#   keras_layer,
#   tf.keras.layers.Dense(num_classes, activation="softmax")
# ])
# 
# build the model with input image shape as (64, 64, 3)
# 
# m.build([None, 64, 64, 3])
# m.compile(
#     loss="categorical_crossentropy", 
#     optimizer="adam", 
#     metrics=["accuracy", tfa.metrics.F1Score(num_classes)]
# )
# m.summary()

# Step 9: Fine-Tuning The Model

# In[ ]:


model_name = "satellite-classification"
model_path = os.path.join("results", model_name + ".h5")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)

# set the training & validation steps since we're using .repeat() on our dataset
# number of training steps
n_training_steps   = int(num_examples * 0.6) // batch_size
# number of validation steps
n_validation_steps = int(num_examples * 0.2) // batch_size

# train the model
history = m.fit(
    train_ds, validation_data=valid_ds,
    steps_per_epoch=n_training_steps,
    validation_steps=n_validation_steps,
    verbose=1, epochs=5, 
    callbacks=[model_checkpoint]
)


# Step 10: Model Evaluation

# In[ ]:


# load the best weights
m.load_weights(model_path)

# number of testing steps
n_testing_steps = int(all_ds[1].splits["train"].num_examples * 0.2)
# get all testing images as NumPy array
images = np.array([ d["image"] for d in test_ds.take(n_testing_steps) ])
print("images.shape:", images.shape)
# get all testing labels as NumPy array
labels = np.array([ d["label"] for d in test_ds.take(n_testing_steps) ])
print("labels.shape:", labels.shape)

# feed the images to get predictions
predictions = m.predict(images)
# perform argmax to get class index
predictions = np.argmax(predictions, axis=1)
print("predictions.shape:", predictions.shape)

from sklearn.metrics import f1_score

accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(labels, predictions)
print("Accuracy:", accuracy.result().numpy())
print("F1 Score:", f1_score(labels, predictions, average="macro"))


# Output:
# 
# Accuracy: 0.9677778
# F1 Score: 0.9655686619720163
