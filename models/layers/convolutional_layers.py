import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.activation = tf.keras.layers.Activation(activation) if activation else None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv2D, self).get_config()
        config.update({
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'strides': self.conv.strides,
            'padding': self.conv.padding,
            'activation': self.activation,
        })
        return config
