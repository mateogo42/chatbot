from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pickle
from typing import List
from cli import preprocess_input
from pathlib import Path
from model import Chatbot
import numpy as np
import requests
import os
import json

TOKEN = os.environ.get("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.environ.get('PAGE_ACCESS_TOKEN')

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

class WebhookData(BaseModel):
    object: str = ""
    entry: List = []


@app.router.get("/")
async def verify(request: Request):

    if request.query_params.get("hub.mode") == "subscribe" and request.query_params.get("hub.challenge"):
        if not request.query_params.get("hub.verify_token") == TOKEN:
            return Response(content="Los tokens no coinciden", status_code=403)
        return Response(content=request.query_params["hub.challenge"])

    return Response(content="Faltan argumentos en el request", status_code=400)



@app.post("/")
async def chat(webhook_data: WebhookData):

    if webhook_data.object == 'page':
        for entry in webhook_data.entry:
            for event in entry['messaging']:
                if event.get('message', None):
                    sender_id = event['sender']['id']
                    recipient_id = event['recipient']['id']
                    text = event['message']['text']
                    answer = create_response(text)
                    send_message(recipient_id, answer)

    

def create_response(text: str) -> str:
    question = preprocess_input(text) 
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

    return answer

async def send_message(recipient_id: str, message: str):
    params = {'access-token': PAGE_ACCESS_TOKEN}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({
        'recipient': {
            'id': recipient_id
        },
        'message': {
            'text': message
        }
    })

    response = requests.post('https://graph.facebook.com/v2.6/me/messages', params=params, headers=headers, data=data)




