import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
import copy
import random
from pretraining_functions import custom_augment,resnet50_encoder, WarmUpCosine, BarlowTwins, parse_images, load_image

# hyper params

AUTOTUNE = tf.data.AUTOTUNE
SEED = 123
input_shape = (512, 512, 3)
width = input_shape[0]
batch_size = 16
model_path = os.path.join("..", "..", "..", "media", "FS2","models")

# data loading

image_dir = os.path.join("..", "..", "..", "media", "FS2", "data", "pretraining_data", "png")
for subdir, dirs, files in os.walk(image_dir):
    new_files = [os.path.join(image_dir, file) for file in files]
ds = tf.data.Dataset.from_tensor_slices(new_files).map(lambda x: load_image(x))
ds = ds.flat_map(lambda x:  tf.data.Dataset.from_tensor_slices(parse_images(x, input_shape[0:2], 16)))

# data augmentation

ssl_ds_one = (
    ds.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

#ssl_ds_two = ds2 #tf.data.Dataset.from_tensor_slices(X_train)
ssl_ds_two = (
    ds.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# encoder - resnet

input = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))
encoder = resnet50_encoder(input)
encoder.summary()

# BT

PROJECT_DIM = width/2
batch_size = 4
EPOCHS =25
WEIGHT_DECAY = 5e-4

STEPS_PER_EPOCH = len(new_files)*24 // batch_size
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
WARMUP_EPOCHS = int(EPOCHS * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

lr_decayed_fn = WarmUpCosine(
    learning_rate_base=1e-4,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=0.0,
    warmup_steps=WARMUP_STEPS
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=2,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

barlow_twins = BarlowTwins(encoder)
barlow_twins.compile(optimizer=optimizer)
history = barlow_twins.fit(ssl_ds, 
                           epochs=EPOCHS,
                           callbacks=[early_stop_cb#, model_ckpt_cb
                                      ])

plt.plot(history.history["loss"])
plt.grid()
plt.title("Barlow Twin Loss")
plt.savefig("BT_resnet_loss.pdf")

barlow_twins.encoder.save(os.path.join(model_path, "barlow_twins_resnet"))