import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Input, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization, ReLU, Dropout
from tensorflow.keras import Model

############################
#
# Encoder
#
############################
def conv_layers(filters_list):
    conv_layers = tf.keras.Sequential()
    for filter in filters_list:
        conv_layers.add(Conv2D(filter, kernel_size=3, strides=(1, 1), padding='same'))
        conv_layers.add(BatchNormalization())
        conv_layers.add(ReLU())
        conv_layers.add(Dropout(rate=0.2))
    return conv_layers


class TfAbs(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return tf.math.abs(x)


class Encoder(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.conv1 = conv_layers([16, 16])
    self.conv2 = conv_layers([32, 32])
    self.conv3 = conv_layers([64, 64, 64])
    self.conv4 = conv_layers([128, 128, 128])

  def _process_input(self, input):
    x1 = self.conv1(input)
    x1_max =  MaxPool2D(pool_size=(2, 2), strides=2)(x1)

    x2 = self.conv2(x1_max)
    x2_max =  MaxPool2D(pool_size=(2, 2), strides=2)(x2)

    x3 = self.conv3(x2_max)
    x3_max =  MaxPool2D(pool_size=(2, 2), strides=2)(x3)

    x4 = self.conv4(x3_max)
    x4_max =  MaxPool2D(pool_size=(2, 2), strides=2)(x4)

    return x1, x2, x3, x4, x4_max
  
  def call(self, input1, input2):
    encoded_1 = self._process_input(input1)
    encoded_2 = self._process_input(input2)
    diff_encoded = tuple([TfAbs()(p1 - p2) for (p1, p2) in zip(encoded_1[:-1], encoded_2[:-1])])


    return diff_encoded, encoded_1[-1], encoded_2[-1]


############################
#
# Decoder
#
############################

class Sigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        return tf.math.sigmoid(x)


def conv_transpose_layers(filters_list, strides=(1,1)):
    convt_layers = tf.keras.Sequential()
    for filter in filters_list:
        convt_layers.add(Conv2DTranspose(filter, kernel_size=3, strides=strides, padding='same'))
        convt_layers.add(BatchNormalization())
        convt_layers.add(ReLU())
        convt_layers.add(Dropout(rate=0.2))
    return convt_layers

def decoder(input_1, diff_input):
    x5 = input_1
    x1, x2, x3, x4 = diff_input
    x = conv_transpose_layers([128], (2,2))(x5)
    
    x = Concatenate(axis=-1)([x, x4])
    x = conv_transpose_layers([128, 128, 64])(x)
    x = conv_transpose_layers([64], (2,2))(x)

    x = Concatenate(axis=-1)([x, x3])
    x = conv_transpose_layers([64, 64, 32])(x)
    x = conv_transpose_layers([32], (2,2))(x)

    x = Concatenate(axis=-1)([x, x2])
    x = conv_transpose_layers([32, 16])(x)
    x = conv_transpose_layers([16], (2,2))(x)

    x = Concatenate(axis=-1)([x, x1])
    x = conv_transpose_layers([16])(x)
    x = Conv2DTranspose(1, kernel_size=3, strides=(1, 1), padding='same')(x)

    x = Sigmoid()(x)
    
    return x


############################
#
# Model
#
############################

def get_siamese_model(IMG_SIZE):
    input1 = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    input2 = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    encoder = Encoder()
    
    diffs, last_max_encoder1, last_max_encoder2 = encoder(input1, input2)

    decoder_output = decoder(last_max_encoder1, diffs)

    model = Model(inputs=[input1, input2], outputs=[decoder_output])
    return model
