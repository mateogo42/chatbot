from pathlib import Path
from unidecode import unidecode
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from model import Chatbot
import pickle
from termcolor import colored
import sys


data_dir = Path('data')
output_dir = Path('model/model.hdf5')
START_TOKEN = '<START> '
END_TOKEN = ' <END>'
with open(data_dir/'tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

tokenizer = tokenizer_data['tokenizer']
vocab_size = tokenizer_data['vocab_size']
max_input_length = tokenizer_data['max_input_length']
max_output_length = tokenizer_data['max_output_length']

def preprocess_input(text: str):
    text = text.lower()
    text = unidecode(text)
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_input_length, padding='post')

    return padded_text

def chat():
    enc_input, dec_input, dec_target = tokenizer_data['model_bootstrap']
    model = Chatbot(vocab_size + 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.train_on_batch([enc_input, dec_input], dec_target)
    model.load_weights(output_dir)

    print(colored("MovieBot:", 'red', attrs=['bold']), "Hola, soy un bot programado para responder preguntas sobre peliculas. ¿En qué te puedo ayudar?. Para salir escriba la palabra adios")

    stop_words = ['adios']
    should_exit = False
    try:
        while not should_exit:
                raw_question = input(">>")
                should_exit = raw_question.strip() in stop_words
                if should_exit: break
                question = preprocess_input(raw_question) 
                h, c = model.encoder.predict(question)
                
                stop_condition = False
                answer = ''
                empty_target_seq = np.array([[tokenizer.word_index['start']]])

                while not stop_condition:
                    decoder_output, h, c = model.decoder.predict([empty_target_seq, h, c])
                    sampled_word_index = np.argmax( decoder_output[0, -1, :] )
                    sampled_word = tokenizer.index_word.get(sampled_word_index, None)
                    if sampled_word == 'end' or len(answer.split()) > max_output_length:
                        stop_condition = True
                    else:
                        answer += f"{sampled_word} "    

                    empty_target_seq = np.array([[sampled_word_index]])

                print(colored("MovieBot:", "red", attrs=['bold']), f"{answer}")
    except KeyboardInterrupt:
        pass
    finally:
        print(colored("MovieBot:", "red", attrs=['bold']), "Adios.")
        sys.exit(0)


if __name__ == '__main__':
    chat()