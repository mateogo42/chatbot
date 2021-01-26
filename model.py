from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Input, Dense


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, embedding_dim: int, n_units: int):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm_1 = LSTM(n_units, return_state=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        output, state_h, state_c = self.lstm_1(x)
        
        return output, state_h, state_c

    def predict(self, inputs):
        x = self.embedding(inputs)
        _, state_h, state_c = self.lstm_1(x)
        
        return state_h, state_c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, n_units):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm_1 = LSTM(n_units, return_state=True, return_sequences=True)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        decoder_input, state_h, state_c = inputs
        x = self.embedding(decoder_input)
        x, _, _ = self.lstm_1(x, initial_state=[state_h, state_c])
        output = self.dense(x)

        return output

    def predict(self, inputs):
        decoder_input, state_h, state_c = inputs
        x = self.embedding(decoder_input)
        x, dec_h, dec_c = self.lstm_1(x, initial_state=[state_h, state_c])
        output = self.dense(x)

        return output, dec_h, dec_c


class Chatbot(tf.keras.models.Model):
    def __init__(self, vocab_size: int):
        super(Chatbot, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(self.vocab_size, 512, 256)
        self.decoder = Decoder(self.vocab_size, 512, 256)

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_output, enc_h, enc_c = self.encoder(encoder_input)
        output = self.decoder([decoder_input, enc_h, enc_c])

        return output

    def chat(self, inputs):
        pass
    
