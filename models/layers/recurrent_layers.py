import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomLSTM(Layer):
    def __init__(self, units, return_sequences=False, return_state=False, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=return_sequences, return_state=return_state)

    def call(self, inputs, initial_state=None):
        return self.lstm(inputs, initial_state=initial_state)

    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        config.update({
            'units': self.lstm.units,
            'return_sequences': self.lstm.return_sequences,
            'return_state': self.lstm.return_state,
        })
        return config
