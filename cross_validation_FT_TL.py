import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
import random
import cv2
import re
from sklearn.metrics import classification_report
import pandas as pd
from finetuning_functions import training_plots, dice_coefficient, Unet, plot_validation_masks, make_augmentations, combine_masks, numerical_sort, DataLoader
from pretraining_functions import unet_mod, Unet_encoder, BarlowTwins, resnet50_encoder
from sklearn.model_selection import KFold
from tqdm import tqdm

# Model parameters

input_size = (512,512)
batch_size = 16
val_batch_size = 16
random.seed(42)
epochs =100
lr = 0.00001
drop = 0.3

# Load data

labeled_data_dir = os.path.join("..", "..", "..", "media", "FS2", "data", "updated_masks")
masked_images = [0 for i in range(150)]
smoke_masks = [0 for i in range(150)] 
cloud_masks =[0 for i in range(150)]
image_nos = []

# loading masks
for subdir, dirs, files in os.walk(labeled_data_dir):
    for file in files:
        file_path = os.path.join(labeled_data_dir, file)
        mask = np.load(os.path.join(labeled_data_dir, file))
        image_no = int(file.split("-")[1])-1
        image_nos.append(image_no)
        if "Smoke" in file_path and not "cloud" in file:
            smoke_masks[image_no] = mask
        if "cloud" in file:
            cloud_masks[image_no] = mask
# correcting masks
for i in range(len(smoke_masks)):
    if np.shape(cloud_masks[i]) == ():
        blank_img = np.zeros(smoke_masks[i].shape,dtype=np.uint8)
        cloud_masks[i] = blank_img
    if np.shape(smoke_masks[i]) == ():
        blank_img = np.zeros(cloud_masks[i].shape,dtype=np.uint8)
        smoke_masks[i] = blank_img

assert(len(smoke_masks) == len(cloud_masks))

# storing masks
masked_images = combine_masks(smoke_masks, cloud_masks)

# loading images
originals = []
original_image_dir = os.path.join("..", "..", "..", "media", "FS2", "data", "updated_images")
for subdir, dirs, files in os.walk(original_image_dir):
    for file in sorted(files, key=numerical_sort):
        image = Image.open(os.path.join(original_image_dir, file)).convert('RGB')
        np_original = np.asarray(image)
        originals.append(np_original)
originals = originals[:150]

assert(len(originals) == len(masked_images))

# Splitting data
random.seed(42)
test_inds = random.sample([i for i in range(150)], 25)
test_dataset = DataLoader(masked_images, originals, target_shape=input_size)
inputs, targets = test_dataset.data_generator(150)
test_input = [inputs[ind] for ind in test_inds]
test_target = [targets[ind] for ind in test_inds]
test = tf.data.Dataset.from_tensor_slices((test_input, test_target))
test_batches = test.batch(val_batch_size)

# define training and cross validation set
inputs = [inputs[ind] for ind in range(150) if ind not in test_inds]
targets = [targets[ind] for ind in range(150) if ind not in test_inds]
#train_val = tf.data.Dataset.from_tensor_slices((inputs, targets))

# setting up folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Model functions
model_path = os.path.join("..", "..", "..", "media", "FS2","models")
# Callbacks
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# model_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
#     "models/smoke_segmentation_256.h5", # update....
#     monitor="val_loss",
#     verbose=0,
#     save_best_only=True,
#     save_weights_only=False,
#     mode="auto",
#     save_freq="epoch",
# )

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=10e-8,
)

trainable = [True, False] # true is for fine tuning, false is for tranfer learning
models = ["unet", "resnet"]
for learn in tqdm(trainable):
    # k fold cross validation
    metric_results = {mod: {"train_metrics":
                         {"accuray":[], "precision":[], "recall":[], "f-1":[], "mean IoU":[]},
                         "test_metrics":
                         {"accuray":[], "precision":[], "recall":[], "f-1":[], "mean IoU":[]},
                         "val_metrics":
                         {"accuray":[], "precision":[], "recall":[], "f-1":[], "mean IoU":[]}
                        }
                        for mod in models}
    for train, val in tqdm(kfold.split(inputs, targets)):
        # set up train and val
        # augmentations on train
        train_inputs, train_labels = [inputs[j] for j in train], [targets[j] for j in train]
        train_inputs, train_labels = make_augmentations(train_inputs, train_labels)
        fold_val = tf.data.Dataset.from_tensor_slices(([inputs[j] for j in val], [targets[j] for j in val]))
        fold_train = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        train_batches = fold_train.batch(batch_size)
        val_batches = fold_val.batch(val_batch_size)
        # set up model
        input = tf.keras.layers.Input(shape=(input_size[0], input_size[1], 3))
        
        for model in tqdm(models):

            if model == "unet":
                optimizer = tf.keras.optimizers.SGD()
                encoder = Unet_encoder(input)
                barlow_twins = BarlowTwins(encoder)
                barlow_twins.compile(optimizer=optimizer)
                barlow_twins.encoder.load_weights(os.path.join(model_path, "barlow_twins_UNET"))
                backbone = tf.keras.Model( barlow_twins.encoder.input, barlow_twins.encoder.layers[-2].output)
                backbone.trainable = learn
                skip_layers = [backbone.get_layer(index=6).output,
                backbone.get_layer(index=14).output,
                backbone.get_layer(index=22).output,
                backbone.get_layer(index=30).output]
                unet = Unet(input, dropout=drop)
            elif model == "resnet":
                optimizer = tf.keras.optimizers.SGD()
                input = tf.keras.layers.Input(shape=(input_size[0], input_size[1], 3))
                encoder = resnet50_encoder(input)
                barlow_twins = BarlowTwins(encoder)
                barlow_twins.compile(optimizer=optimizer)
                backbone = tf.keras.Model( barlow_twins.encoder.input, barlow_twins.encoder.layers[-2].output)
                backbone.trainable = learn
                s1 = backbone.get_layer(index=0).output           ## (512 x 512)
                s2 = backbone.get_layer("conv1_relu").output        ## (256 x 256)
                s3 = backbone.get_layer("conv2_block3_out").output  ## (128 x 128)
                s4 = backbone.get_layer("conv3_block4_out").output  ## (64 x 64)
                skip_layers = [s1, s2, s3, s4]

            unet = unet_mod(backbone, skip_layers, input, num_filters=16, kernel=3, dropout=drop, batch_norm=True, train_encoder=learn)
            unet.compile(
            loss="sparse_categorical_crossentropy", 
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
            metrics=["accuracy", 
                    tf.keras.metrics.MeanIoU(num_classes=3, sparse_y_true=True, sparse_y_pred=False)]
                    )
            # train
            history = unet.fit(
            train_batches,
            validation_data=val_batches,
            batch_size=batch_size,
            epochs=epochs, 
            callbacks=[
                early_stop_cb, 
                # model_ckpt_cb, 
                reduce_lr_cb
            ])
            # store metrics
            m = tf.keras.metrics.MeanIoU(num_classes=3)
            # train metrics
            train_non_aug = tf.data.Dataset.from_tensor_slices(([inputs[j] for j in train], [targets[j] for j in train]))
            train_images, train_labels = tuple(zip(*train_non_aug))
            train_pred = unet.predict(train_non_aug.batch(batch_size))
            train_metrics_k = classification_report(np.array(train_labels).flatten(), np.array(tf.argmax(train_pred, axis=-1)).flatten(), output_dict=True)
            m.update_state(train_labels, tf.argmax(train_pred, axis=-1))
            metric_results[model]["train_metrics"]["mean IoU"].append(m.result().numpy())
            metric_results[model]["train_metrics"]["accuray"].append(train_metrics_k["accuracy"])
            metric_results[model]["train_metrics"]["recall"].append(train_metrics_k["macro avg"]["recall"])
            metric_results[model]["train_metrics"]["precision"].append(train_metrics_k["macro avg"]["precision"])
            metric_results[model]["train_metrics"]["f-1"].append(train_metrics_k["macro avg"]["f1-score"])
            # val metrics
            images, val_labels = tuple(zip(*fold_val))
            val_pred = unet.predict(val_batches)
            val_metrics_k = classification_report(np.array(val_labels).flatten(), np.array(tf.argmax(val_pred, axis=-1)).flatten(), output_dict=True)
            m.update_state(val_labels, tf.argmax(val_pred, axis=-1))
            metric_results[model]["val_metrics"]["mean IoU"].append(m.result().numpy())
            metric_results[model]["val_metrics"]["accuray"].append(val_metrics_k["accuracy"])
            metric_results[model]["val_metrics"]["recall"].append(val_metrics_k["macro avg"]["recall"])
            metric_results[model]["val_metrics"]["precision"].append(val_metrics_k["macro avg"]["precision"])
            metric_results[model]["val_metrics"]["f-1"].append(val_metrics_k["macro avg"]["f1-score"])
            # test metrics
            images, test_labels = tuple(zip(*test))
            test_pred = unet.predict(test_batches)
            test_metrics_k = classification_report(np.array(test_labels).flatten(), np.array(tf.argmax(test_pred, axis=-1)).flatten(), output_dict=True)
            m.update_state(test_labels, tf.argmax(test_pred, axis=-1))
            metric_results[model]["test_metrics"]["mean IoU"].append(m.result().numpy())
            metric_results[model]["test_metrics"]["accuray"].append(test_metrics_k["accuracy"])
            metric_results[model]["test_metrics"]["recall"].append(test_metrics_k["macro avg"]["recall"])
            metric_results[model]["test_metrics"]["precision"].append(test_metrics_k["macro avg"]["precision"])
            metric_results[model]["test_metrics"]["f-1"].append(test_metrics_k["macro avg"]["f1-score"])

    for model in models:
        for dset in metric_results[model]:
            for metric in metric_results[model][dset]:
                metric_results[model][dset][metric].append(np.mean(metric_results[model][dset][metric][:5]))
                metric_results[model][dset][metric].append(np.std(metric_results[model][dset][metric][:5]))
        with pd.ExcelWriter('results/'+model+learn*"_fine_tuned"+(learn==False)*"_transfer"+".xlsx") as writer: 
            for set_metrics in metric_results[model]:
                    pd.DataFrame(metric_results[model][set_metrics]).to_excel(writer, set_metrics)
