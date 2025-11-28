import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
import random
from tensorflow.keras.applications import ResNet50

AUTOTUNE = tf.data.AUTOTUNE
SEED = 123
image_shape = (1500, 2500, 5)

### Data processing

def split(img, target_shape, overlap):
    sh = list(img.shape)
    sh[0], sh[1] = sh[0] + (overlap[0] * 2), sh[1] + (overlap[1] * 2)
    splitted = []
    v_stride = target_shape[0]
    h_stride = target_shape[1]
    v_step = target_shape[0] + 2 * overlap[0]
    h_step = target_shape[1] + 2 * overlap[1]
    nrows, ncols = max((img.shape[0] - overlap[0]*2) // target_shape[0], 1), max((img.shape[1] - overlap[1]*2) // target_shape[1], 1)
    for i in range(nrows):
        for j in range(ncols):
            h_start = j*h_stride
            v_start = i*v_stride
            cropped = img[v_start:v_start+v_step, h_start:h_start+h_step]
            splitted.append(cropped)
    return splitted, (nrows, ncols)

def get_tiles(image, target_shape, overlap=0):
    padding_tuple = ((0,0), (0,0), (0,0)) 
    padded_image = image
    overlap = [overlap, overlap]
    if target_shape == image.shape[:2]:
        images = [image]
        full_tiled_shape = (1,1)
    else:
        target_shape = list(target_shape)
        if target_shape[0] < image.shape[0]:
            target_shape[0] = target_shape[0]-(2*overlap[0])
        elif target_shape[0] >= image.shape[0]:
            overlap[0] = 0
        if target_shape[1] < image.shape[1]:
            target_shape[1] = target_shape[1]-(2*overlap[1])
        elif target_shape[1] >= image.shape[1]:
            overlap[1] = 0
        target_shape = tuple(target_shape)
        total_width = (target_shape[0]*np.ceil(image.shape[0]/(target_shape[0]))) + (overlap[0]*2) # *2
        total_height = (target_shape[1]*np.ceil(image.shape[1]/(target_shape[1]))) + (overlap[1]*2) # *2
        remainders = (total_width - image.shape[0], total_height - image.shape[1])
        padding_tuple = ((0, int(remainders[0])), (0, int(remainders[1])), (0,0))
        padded_image = tf.pad(image, padding_tuple, 'constant', constant_values=(0))
        tiled_array, full_tiled_shape = split(padded_image, target_shape, overlap)
        images = tiled_array
    return images, full_tiled_shape, padding_tuple, overlap

def parse_images(image, target_shape, overlap): #add tilling here
        images, _, _, _ = get_tiles(image, target_shape, overlap)
        return images

def load_helper(file):
    image = np.asarray(Image.open(file.numpy()).convert('RGB'))
    return image

def load_image(file):
    image = tf.py_function(load_helper, [file], tf.uint8)
    image.set_shape(image_shape)
    return image


### Unet 
def Conv2dBlock(input_layer, num_filters, kernel=3, batch_norm=True):
    # first convolution
    x = tf.keras.layers.Conv2D(filters=num_filters, 
                               kernel_size=(kernel, kernel), 
                               kernel_initializer='he_normal', 
                               padding='same')(input_layer)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # second convolution
    x = tf.keras.layers.Conv2D(filters=num_filters, 
                               kernel_size=(kernel, kernel), 
                               kernel_initializer='he_normal', 
                               padding='same')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def DecoderBlock(input_layer, skip_connection, num_filters, kernel=3, batch_norm=True, dropout=0.1):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (kernel, kernel), strides = (2, 2), padding = 'same')(input_layer)
    x = tf.keras.layers.concatenate([x, skip_connection])
    x = tf.keras.layers.Dropout(dropout)(x)
    x = Conv2dBlock(x, num_filters, kernel=kernel, batch_norm=batch_norm)
    return x

def Unet_encoder(input, num_filters=16, dropout=0.1, batch_norm=True):
    # Contraction phase - includes 4 rounds of convolution block -> max pooling
    # round 1
    c1 = Conv2dBlock(input, num_filters*1, kernel=3, batch_norm=batch_norm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)
    # round 2
    c2 = Conv2dBlock(p1, num_filters*2, kernel=3,batch_norm=batch_norm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)
    # round 3
    c3 = Conv2dBlock(p2, num_filters*4, kernel=3, batch_norm=batch_norm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)
    # round 4
    c4 = Conv2dBlock(p3, num_filters*8, kernel=3, batch_norm=batch_norm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)

    # Bottle neck phase - expands the feature dimension
    c5 = Conv2dBlock(p4, num_filters*16, kernel=3, batch_norm=batch_norm)

    out = tf.keras.layers.GlobalMaxPooling2D('channels_last')(c5)
    # output
    # output = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(c5)
    model = tf.keras.Model(inputs=[input], outputs=[out])
    return model

def unet_mod(encoder, skip_layers, inputs, num_filters=16, kernel=3, dropout=0.1, batch_norm=True, train_encoder=True):
    # define encoder
    encoder.trainable = train_encoder
    x = encoder.output
    d1 = DecoderBlock(x, skip_layers[3], num_filters*8, kernel, batch_norm, dropout)                     ## (64 x 64)
    d2 = DecoderBlock(d1, skip_layers[2], num_filters*4 ,kernel, batch_norm, dropout)                     ## (128 x 128)
    d3 = DecoderBlock(d2, skip_layers[1], num_filters*2, kernel, batch_norm, dropout)                     ## (256 x 256)
    d4 = DecoderBlock(d3, skip_layers[0], num_filters*1, kernel, batch_norm, dropout)                      ## (512 x 512)
    
    # output layer - 4 classes 
    output = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(d4)
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    return model

### Resnet50 encoder for Unet
def resnet50_encoder(inputs):
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs, pooling="max")
    s1 = resnet50.get_layer(index=0).output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)
    b1 = resnet50.get_layer("conv4_block6_out").output
    out = tf.keras.layers.GlobalMaxPooling2D('channels_last')(b1)
    encoder = tf.keras.Model(inputs=[inputs], outputs=[out])
    return encoder

### Warmup
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

### BT augmentation
def random_resize_crop(image, scale=[0.75, 1.0], crop_size=128):
    if crop_size == 32:
        image_shape = 48
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = 96
        image = tf.image.resize(image, (image_shape, image_shape))
    size = tf.random.uniform(
        shape=(1,),
        minval=scale[0] * image_shape,
        maxval=scale[1] * image_shape,
        dtype=tf.float32,
    )
    size = tf.cast(size, tf.int32)[0]
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))
    return crop_resize

def flip_random_crop(image):
    image = tf.image.random_flip_left_right(image)
    image = random_resize_crop(image, crop_size=image.shape[0])
    return image

def float_parameter(level, maxval):
    return tf.cast(level * maxval / 10.0, tf.float32)

def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)

def rotation(image):
    augmented_image = tf.image.rot90(image)
    return augmented_image

def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x

def custom_augment(image):
    image = tf.cast(image, tf.float32)
    image = flip_random_crop(image)
    image = random_apply(rotation, image, p=0.5)
    #image = random_apply(color_jitter, image, p=0.9)
    #image = random_apply(color_drop, image, p=0.3)
    #image = random_apply(solarize, image, p=0.3)
    return image

### BT model functions
def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss 

class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd) 

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}



