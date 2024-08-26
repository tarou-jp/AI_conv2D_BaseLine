import os
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.config import config
from models.layers.custom_layers import CustomConv2D, CustomAttention, CustomLSTM
from models.attention_model import build_custom_cnn_lstm_attention_model

class Trainer:
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.checkpoint_dir = config.model_save_path
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compile_model(self, learning_rate=config.learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=config.epochs, batch_size=config.batch_size):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.h5')
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history

    def evaluate(self, test_data):
        loss, accuracy = self.model.evaluate(test_data)
        return loss, accuracy
