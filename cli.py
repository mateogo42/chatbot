from pathlib import Path
from model import Chatbot
from termcolor import colored
import sys


tokenizer_path = Path("data/tokenizer.pkl")
weights_path = Path("model/model.hdf5")
model = Chatbot.from_files(tokenizer_path, weights_path)

def chat():

    print(colored("MovieBot:", 'red', attrs=['bold']), "Hola, soy un bot programado para responder preguntas sobre peliculas. ¿En qué te puedo ayudar?. Para salir escribe la palabra adios")

    stop_words = ['adios']
    should_exit = False
    try:
        while not should_exit:
            raw_question = input(">>")
            should_exit = raw_question.strip() in stop_words
            if should_exit: break
            answer = model.chat(raw_question)
            print(colored("MovieBot:", "red", attrs=['bold']), f"{answer}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        print(colored("MovieBot:", "red", attrs=['bold']), "Adios.")
        sys.exit(0)


if __name__ == '__main__':
    chat()
