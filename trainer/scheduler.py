import tensorflow as tf
from utils.config import config

def get_scheduler(initial_lr=config.learning_rate):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    return lr_schedule
