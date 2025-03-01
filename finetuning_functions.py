import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import re
import random
import cv2

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts 

def combine_masks(smoke_masks, cloud_masks):
    masks = [0 for i in range(len(smoke_masks))]
    for i in range(len(smoke_masks)):
        smoke = smoke_masks[i]
        smoke = np.where(smoke==0, smoke, 1)
        temp = cloud_masks[i]
        temp = np.where(cloud_masks[i]!=255, cloud_masks[i], 2) #changing color to grey
        temp = np.where(cloud_masks[i]==0, temp, 2)
        full_mask = temp + smoke
        full_mask = np.where(full_mask!=3, full_mask, 1 )
        masks[i] = full_mask
    return masks

def dice_coefficient(true, pred):
    smooth = 1.
    true_flat = K.flatten(true)
    pred_flat = K.flatten(pred)
    intersection = K.sum(true * pred)
    score = (2. * intersection + smooth) / (K.sum(true_flat) + K.sum(pred_flat) + smooth)
    return score

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1) 
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def convert_mask_to_colors(mask):
    mapping = {0: 0, 1: 127.5, 2: 225}
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
    mapping_ar[k] = v
    mask = mapping_ar[mask.astype(int)]
    return mask

def training_plots(history):
    # Learning Rate
    plt.plot(history.history['lr'])
    plt.title('Learning Rate')
    plt.xlabel("Epochs")
    plt.ylabel("LR Value")
    plt.show()

    # Loss Curves
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title('Loss Curves')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Accuracy Curves
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.title('Accuracy Curves')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # IoU Curves
    plt.plot(history.history['mean_io_u'], label="Train IoU")
    plt.plot(history.history['val_mean_io_u'], label="Val IoU")
    plt.title('IoU Curves')
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()
    plt.show()


def plot_validation_masks(inputs, preds, targets):
    plt.rcParams["figure.figsize"] = (20, 5)

    for i, p, t in zip(inputs, preds, targets):
        p = create_mask(p)
        plt.subplot(1, 3, 1)
        plt.imshow(i)
        plt.title("Input Image")
        plt.axis("off")
    
        plt.subplot(1, 3, 2)
        plt.imshow(convert_mask_to_colors(p), cmap="gray")
        plt.title("Predicted Mask",)
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(convert_mask_to_colors(t), cmap="gray")
        plt.title("True Mask")
        plt.axis("off")
        plt.show()

def Conv2dBlock(input_layer, num_filters, kernel=3, BatchNorm=True):
    # first convolution
    x = tf.keras.layers.Conv2D(filters=num_filters, 
                               kernel_size=(kernel, kernel), 
                               kernel_initializer='he_normal', 
                               padding='same')(input_layer)
    if BatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # second convolution
    x = tf.keras.layers.Conv2D(filters=num_filters, 
                               kernel_size=(kernel, kernel), 
                               kernel_initializer='he_normal', 
                               padding='same')(x)
    if BatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Unet(input, num_filters=16, dropout=0.1, BatchNorm=True):
    # Contraction phase - includes 4 rounds of convolution block -> max pooling
    # round 1
    c1 = Conv2dBlock(input, num_filters*1, kernel=3, BatchNorm=BatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)
    # round 2
    c2 = Conv2dBlock(p1, num_filters*2, kernel=3, BatchNorm=BatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)
    # round 3
    c3 = Conv2dBlock(p2, num_filters*4, kernel=3, BatchNorm=BatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)
    # round 4
    c4 = Conv2dBlock(p3, num_filters*8, kernel=3, BatchNorm=BatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)

    # Bottle neck phase - expands the feature dimension
    c5 = Conv2dBlock(p4, num_filters*16, kernel=3, BatchNorm=BatchNorm)
    
    # Expansion phase - includes 4 rounds of upsampling -> concatenation -> convolution block
    # round 1
    u6 = tf.keras.layers.Conv2DTranspose(num_filters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = Conv2dBlock(u6, num_filters * 8, kernel=3, BatchNorm=BatchNorm)
    # round 2
    u7 = tf.keras.layers.Conv2DTranspose(num_filters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = Conv2dBlock(u7, num_filters * 4, kernel=3, BatchNorm=BatchNorm)
    # round 3
    u8 = tf.keras.layers.Conv2DTranspose(num_filters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = Conv2dBlock(u8, num_filters * 2, kernel=3, BatchNorm=BatchNorm)
    # round 4
    u9 = tf.keras.layers.Conv2DTranspose(num_filters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = Conv2dBlock(u9, num_filters * 1, kernel=3, BatchNorm=BatchNorm)
    
    # output layer - 3 classes 
    output = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(c9)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model

def make_augmentations(inputs, targets):
    # augmentation - flip hz, flip vt, hz+vt flip, rotation 90, 180, 270

    augmented_inputs = []
    augmented_labels = []

    # flip hz
    hz_inputs = [np.flip(img, axis=1) for img in inputs]
    hz_labels =  [np.flip(img, axis=1) for img in targets]
    augmented_inputs = hz_inputs
    augmented_labels = hz_labels

    # flip vt
    vt_inputs = [np.flip(img, axis=0) for img in inputs]
    vt_labels =  [np.flip(img, axis=0) for img in targets]
    augmented_inputs = np.append(vt_inputs, augmented_inputs, axis=0)
    augmented_labels = np.append(vt_labels, augmented_labels, axis=0)

    # hz+vt flip
    hzvt_inputs = [np.flip(img, axis=0) for img in hz_inputs]
    hzvt_labels =  [np.flip(img, axis=0) for img in hz_labels]
    augmented_inputs = np.append(hzvt_inputs, augmented_inputs, axis=0)
    augmented_labels = np.append(hzvt_labels, augmented_labels, axis=0)

    # add to originals
    augmented_inputs = np.append(inputs, augmented_inputs, axis=0)
    augmented_labels = np.append(targets, augmented_labels, axis=0)

    # rotations 
    rot90_inputs = [np.flip(img, axis=0) for img in augmented_inputs]
    rot90_labels =  [np.flip(img, axis=0) for img in augmented_labels]
    rot180_inputs = [np.flip(img, axis=0) for img in rot90_inputs]
    rot180_labels =  [np.flip(img, axis=0) for img in rot90_labels]
    rot270_inputs = [np.flip(img, axis=0) for img in rot180_inputs]
    rot270_labels =  [np.flip(img, axis=0) for img in rot180_labels]

    rotation_inputs = rot90_inputs + rot180_inputs + rot270_inputs
    rotation_labels = rot90_labels + rot180_labels + rot270_labels 

    # add rotations
    augmented_inputs = np.append(rotation_inputs, augmented_inputs, axis=0)
    augmented_labels = np.append(rotation_labels, augmented_labels, axis=0)
    return  augmented_inputs, augmented_labels

class DataLoader:
    def __init__(self, masks, originals, target_shape=(256, 256)):
        self.masks = masks
        self.originals = originals
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.target_shape = target_shape
        
    @tf.function
    def parse_images(self, ind):
        mask = cv2.resize(np.expand_dims(self.masks[ind], axis=-1), self.target_shape)
        image = cv2.resize(self.originals[ind], self.target_shape)
        return image, mask
    
    @tf.function
    def data_generator(self, batch_size=4):
        random.seed(0)
        inds = random.sample([i for i in range(len(self.originals))], batch_size)
        inputs = [self.parse_images(ind)[0] for ind in inds]
        targets = [self.parse_images(ind)[1] for ind in inds]
        return inputs, targets

    def data_processor(self):
        inputs = [ cv2.resize(self.originals[ind], self.target_shape) for ind in range(len(self.originals))]
        return inputs