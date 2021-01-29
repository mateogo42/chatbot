from pathlib import Path
import pickle
from unidecode import unidecode
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_units: int):
        super(Encoder, self).__init__()
        self.lstm_1 = LSTM(n_units, return_state=True)

    def call(self, inputs):
        x = inputs
        output, state_h, state_c = self.lstm_1(x)
        
        return output, state_h, state_c

    def predict(self, inputs):
        x = inputs
        _, state_h, state_c = self.lstm_1(x)
        
        return state_h, state_c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, n_units):
        super(Decoder, self).__init__()
        self.lstm_1 = LSTM(n_units, return_state=True, return_sequences=True)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        decoder_input, state_h, state_c = inputs
        x, _, _ = self.lstm_1(decoder_input, initial_state=[state_h, state_c])
        output = self.dense(x)

        return output

    def predict(self, inputs):
        decoder_input, state_h, state_c = inputs
        x, dec_h, dec_c = self.lstm_1(decoder_input, initial_state=[state_h, state_c])
        output = self.dense(x)

        return output, dec_h, dec_c


class Chatbot(tf.keras.models.Model):
    def __init__(self, vocab_size: int, tokenizer, max_input_length: int, 
                        max_output_length: int, embedding_dim: int = 512):
        super(Chatbot, self).__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.encoder = Encoder(256)
        self.decoder = Decoder(vocab_size, 256)

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_input = self.embedding(encoder_input)
        decoder_input = self.embedding(decoder_input)
        encoder_output, enc_h, enc_c = self.encoder(encoder_input)
        output = self.decoder([decoder_input, enc_h, enc_c])

        return output

    def preprocess_text(self, text: str):
        text = text.lower()
        text = unidecode(text)
        tokenized_text = self.tokenizer.texts_to_sequences([text])
        padded_text = pad_sequences(tokenized_text, maxlen=self.max_input_length, padding='post')

        return padded_text

    def chat(self, text: str):
        question = self.preprocess_text(text)
        encoder_input = self.embedding(question)
        h, c = self.encoder.predict(encoder_input)
        stop_condition = False
        answer = ''
        empty_target_seq = np.array([[self.tokenizer.word_index['start']]])

        while not stop_condition:
            decoder_input = self.embedding(empty_target_seq)
            decoder_output, h, c = self.decoder.predict([decoder_input, h, c])
            sampled_word_index = np.argmax(decoder_output[0, -1, :])
            sampled_word = self.tokenizer.index_word.get(sampled_word_index, None)
            if sampled_word == 'end' or len(answer.split()) > self.max_output_length:
                stop_condition = True
            else:
                answer += f"{sampled_word} "    

            empty_target_seq = np.array([np.append(empty_target_seq[0],sampled_word_index)])

        return answer

    @classmethod
    def from_files(cls, tokenizer_path: Path, weights_path: Path):

        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)

        tokenizer = tokenizer_data['tokenizer']
        vocab_size = tokenizer_data['vocab_size']
        max_input_length = tokenizer_data['max_input_length']
        max_output_length = tokenizer_data['max_output_length']

        enc_input, dec_input, dec_target = tokenizer_data['model_bootstrap']
        model = cls(vocab_size + 1, tokenizer=tokenizer, max_input_length=max_input_length, max_output_length=max_output_length)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.train_on_batch([enc_input, dec_input], dec_target)
        print("[*] Loading pretrained model weights...")
        model.load_weights(weights_path)
        print("[*] Finished loading model weights")

        return model