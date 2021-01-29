from pathlib import Path
from model import Chatbot
from termcolor import colored
import sys
import re


tokenizer_path = Path("data/tokenizer.pkl")
weights_path = Path("model/model.hdf5")
model = Chatbot.from_files(tokenizer_path, weights_path)
stop_words = ['adios', 'chao', 'gracias']

def chat():

    print(colored("MovieBot:", 'red', attrs=['bold']), "Hola, soy un bot programado para responder preguntas sobre peliculas. ¿En qué te puedo ayudar?. Para salir escribe la palabra adios")
    should_exit = False
    try:
        while not should_exit:
            text = input(colored("Tu: ", "blue", attrs=['bold']))
            if any([re.match(w, text.lower()) for w in stop_words]):
                print(colored("MovieBot:", "red", attrs=['bold']),
                "Ha sido un placer ayudarte.")
                should_exit = True
            else:
                answer = model.chat(text)
                print(colored("MovieBot:", "red", attrs=['bold']), f"{answer}")
    except KeyboardInterrupt:
        print("\n")
    except Exception as e:
        print(e)
    finally:
        print(colored("MovieBot:", "red", attrs=['bold']), "Adios.")
        sys.exit(0)


if __name__ == '__main__':
    chat()
