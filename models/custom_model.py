from utils.config import config
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from models.layers.custom_layers import CustomConv2D, CustomAttention, CustomLSTM  # カスタムレイヤーのインポート

class CustomModel:
    def __init__(self):
        self.input_shape = config.input_shape
        self.num_labels = config.num_labels
        self.lstm_units = config.lstm_units
        self.num_conv_filters = config.num_conv_filters
        self.attention_units = config.attention_units
        self.batch_size = config.batch_size
        self.model = self.build_model()

    def build_model(self):
        # CNNの入力
        cnn_inputs = Input(shape=self.input_shape, batch_size=self.batch_size, name='cnn_inputs')

        # CustomConv2D層
        cnn_layer = CustomConv2D(filters=self.num_conv_filters, kernel_size=(1, self.input_shape[1]), strides=(1, 1), padding='valid')(cnn_inputs)
        
        # CNN出力の次元を削減
        sq_layer_out = Lambda(lambda x: tf.squeeze(x, axis=2))(cnn_layer)

        # CustomLSTM層
        rnn_layer_output = CustomLSTM(units=self.lstm_units, return_sequences=True)(sq_layer_out)

        # CustomAttention層
        context_vector, attention_weights = CustomAttention(units=self.attention_units)(rnn_layer_output, rnn_layer_output)

        # 出力層
        dense_layer_output = Dense(self.num_labels, activation='softmax')(context_vector)

        # モデルの定義
        model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None):
        if epochs is None:
            epochs = config.epochs
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=self.batch_size)

if __name__ == "__main__":
    # モデルの構築
    custom_model = CustomModel()

    # モデルの概要を表示
    custom_model.summary()

    # トレーニングデータとバリデーションデータを指定してモデルを訓練できます。
    # custom_model.train(X_train, y_train, X_val, y_val)
