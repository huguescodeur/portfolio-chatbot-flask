import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import numpy as np
import json
import random

import tensorflow as tf
from flask import Flask, render_template, request
from flask_cors import CORS

nltk.download('popular')

# Initialiser le lemmatizer de NLTK
lemmatizer = WordNetLemmatizer()

# Charger le modèle
model = tf.keras.models.load_model('model.keras')

# Charger les données prétraitées avec joblib
words = joblib.load('texts.joblib')
classes = joblib.load('labels.joblib')

# Charger les données des intentions depuis le fichier JSON avec encodage UTF-8
with open('data.json', encoding='utf-8') as file:
    intents = json.load(file)


# Fonction pour nettoyer une phrase
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Fonction pour créer un bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# Fonction pour prédire la classe d'une phrase
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Fonction pour obtenir une réponse basée sur l'intention
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Fonction pour obtenir la réponse du chatbot
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# Configurer l'application Flask
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
    
    
    
# {
#       "tag": "not_understood",
#       "patterns": [],
#       "responses": [
#         "Désolé, je ne vous comprends pas!",
#         "Veuillez me donner plus d'informations",
#         "Je ne suis pas sûr de comprendre.",
#         "J'arrive pas à vous suivre!",
#         "Dsl, mais je n'ai pas ces réponses!",
#         "Désolé, je ne comprends pas ce que vous dites.",
#         "Je n'ai pas bien compris votre message.",
#         "Pouvez-vous reformuler votre question ?"
#       ],
#       "context": [""]
#     },


#   {
#        "tag": "experience professionlle",
#        "patterns": [
#          "Parlez-moi de votre expérience professionnelle",
#          "Quels sont vos précédents emplois?",
#          "Quelles entreprises avez vous travaillé?",
#          "Hey",
#          "Hello",
#          "Hi",
#          "Comment ça va?",
#          "Bonjour comment vas tu?"
#        ],
#        "responses": [
#          "Bonjour, comment pourrais-je vous être utile?",
#          "Salut, comment puis-je vous aider?",
#          "Bonjour, comment puis-je vous aider ?"
#        ],
#        "context": [""]
#      }