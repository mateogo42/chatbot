from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from cli import preprocess_input
from pathlib import Path
from model import Chatbot
import numpy as np

with open(Path("data/tokenizer.pkl"), 'rb') as f:
    tokenizer_data = pickle.load(f)

tokenizer = tokenizer_data['tokenizer']
vocab_size = tokenizer_data['vocab_size']
max_input_length = tokenizer_data['max_input_length']
max_output_length = tokenizer_data['max_output_length']

enc_input, dec_input, dec_target = tokenizer_data['model_bootstrap']
model = Chatbot(vocab_size + 1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.train_on_batch([enc_input, dec_input], dec_target)
print("[*] Loading pretrained model weights...")
model.load_weights(Path('model/model.hdf5'))
print("[*] Finished loading model weights")

app = FastAPI()

class Message(BaseModel):
    text: str


@app.post("/chat")
def chat(message: Message):

    question = preprocess_input(message.text) 
    stop_condition = False
    h, c = model.encoder.predict(question)
    empty_target_seq = np.array([[tokenizer.word_index['start']]])
    answer = ''
    while not stop_condition:
        decoder_output, h, c = model.decoder.predict([empty_target_seq, h, c])
        sampled_word_index = np.argmax( decoder_output[0, -1, :] )
        sampled_word = tokenizer.index_word.get(sampled_word_index, None)
        if sampled_word == 'end' or len(answer.split()) > max_output_length:
            stop_condition = True
        else:
            answer += f"{sampled_word} "    

        empty_target_seq = np.array([[sampled_word_index]])

    return {
        'response': answer
    }
    


