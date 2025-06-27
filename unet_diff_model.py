import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Input, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization, ReLU, Dropout
from tensorflow.keras import Model

def conv_layers(filters_list):
    conv_layers = tf.keras.Sequential()
    for filter in filters_list:
        conv_layers.add(Conv2D(filter, kernel_size=3, strides=(1, 1), padding='same'))
        conv_layers.add(BatchNormalization())
        conv_layers.add(ReLU())
        conv_layers.add(Dropout(rate=0.2))
    return conv_layers


def encoder(input1, input2):
    x = Concatenate(axis=-1)([input1, input2])
    x1 = conv_layers([16, 16])(x)
    x1_max = MaxPool2D(pool_size=(2, 2), strides=2)(x1)
    
    x2 = conv_layers([32, 32])(x1_max)
    x2_max = MaxPool2D(pool_size=(2, 2), strides=2)(x2)
    
    x3 = conv_layers([64, 64, 64])(x2_max)
    x3_max = MaxPool2D(pool_size=(2, 2), strides=2)(x3)
    
    x4 = conv_layers([128, 128, 128])(x3_max)
    x4_max = MaxPool2D(pool_size=(2, 2), strides=2)(x4)
    return x1, x2, x3, x4, x4_max


class Sigmoid(tf.keras.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        return tf.math.sigmoid(x)

# class PadLayer(tf.keras.Layer):
#     def __init__(self):
#         super().__init__()
        
#     def call(self, x, x_encoder):
#         return tf.pad(x_encoder,[[0,0],[0,x.shape[1] - x_encoder.shape[1]], [0,x.shape[2] - x_encoder.shape[2]], [0,0]],"SYMMETRIC")

def conv_transpose_layers(filters_list, strides=(1,1)):
    convt_layers = tf.keras.Sequential()
    for filter in filters_list:
        convt_layers.add(Conv2DTranspose(filter, kernel_size=3, strides=strides, padding='same'))
        convt_layers.add(BatchNormalization())
        convt_layers.add(ReLU())
        convt_layers.add(Dropout(rate=0.2))
    return convt_layers

def decoder(input):
    x1, x2, x3, x4, x5 = input
    x = conv_transpose_layers([128], (2,2))(x5)

    # x4 = PadLayer()(x, x4)
    x = Concatenate(axis=-1)([x, x4])
    x = conv_transpose_layers([128, 128, 64])(x)
    x = conv_transpose_layers([64], (2,2))(x)

    # x3 = PadLayer()(x, x3)
    x = Concatenate(axis=-1)([x, x3])
    x = conv_transpose_layers([64, 64, 32])(x)
    x = conv_transpose_layers([32], (2,2))(x)

    # x2 = PadLayer()(x, x2)
    x = Concatenate(axis=-1)([x, x2])
    x = conv_transpose_layers([32, 16])(x)
    x = conv_transpose_layers([16], (2,2))(x)

    # x1 = PadLayer()(x, x1)
    x = Concatenate(axis=-1)([x, x1])
    x = conv_transpose_layers([16])(x)
    x = Conv2DTranspose(1, kernel_size=3, strides=(1, 1), padding='same')(x)

    x = Sigmoid()(x)
    return x

def get_unet_model(IMG_SIZE):
    input1 = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    input2 = Input(shape=(IMG_SIZE,IMG_SIZE,3))

    encoder_output = encoder(input1, input2)
    decoder_output = decoder(encoder_output)

    model = Model(inputs=[input1, input2], outputs=[decoder_output])
    return model
