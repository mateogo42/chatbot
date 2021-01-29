from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import List
from pathlib import Path
from model import Chatbot
import requests
import os
import json
import re

TOKEN = os.environ.get("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.environ.get('PAGE_ACCESS_TOKEN')

tokenizer_path = Path("data/tokenizer.pkl")
weights_path = Path("model/model.hdf5")
model = Chatbot.from_files(tokenizer_path, weights_path)
start_words = ['hola']
stop_words = ['adios', 'chao', 'gracias']

app = FastAPI()

class WebhookData(BaseModel):
    object: str = ""
    entry: List = []


@app.router.get("/")
def verify(request: Request):

    if request.query_params.get("hub.mode") == "subscribe" and request.query_params.get("hub.challenge"):
        if not request.query_params.get("hub.verify_token") == TOKEN:
            return Response(content="Los tokens no coinciden", status_code=403)
        return Response(content=request.query_params["hub.challenge"])

    return Response(content="Faltan argumentos en el request", status_code=400)



@app.post("/")
def chat(webhook_data: WebhookData):
    if webhook_data.object == 'page':
        for entry in webhook_data.entry:
            for event in entry['messaging']:
                if event.get('message', None):
                    sender_id = event['sender']['id']
                    recipient_id = event['recipient']['id']
                    text = event['message']['text']
                    text_words = text.lower().strip().split(' ') 
                    if any([re.match(w, text.lower()) for w in start_words]):
                        send_message(sender_id, "Hola. Soy un bot \U0001F916 programado para responder preguntas sobre peliculas. ¿En qué puedo ayudarte?")
                        if len(text_words) > 1:
                            answer = model.chat(text)
                            send_message(sender_id, answer)
                    elif any([re.match(w, text.lower()) for w in stop_words]):
                        send_message(sender_id, "Ha sido un placer servirte. Si tienes alguna otra pregunta puedes escribirme en cualquier momento \U0001F642")
                    else:
                        answer = model.chat(text)
                        send_message(sender_id, answer)
    return Response(content="ok")

def send_message(recipient_id: str, message: str):
    params = {'access_token': PAGE_ACCESS_TOKEN}
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

    print(response.status_code)




