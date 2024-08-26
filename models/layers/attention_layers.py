import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomAttention(Layer):
    def __init__(self, units, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super(CustomAttention, self).get_config()
        config.update({
            'units': self.units,
        })
        return config
