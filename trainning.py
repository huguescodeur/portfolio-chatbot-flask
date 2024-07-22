import nltk
from nltk.stem import WordNetLemmatizer
import json
import joblib
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Initialiser le lemmatizer de NLTK
lemmatizer = WordNetLemmatizer()

# Charger les données du fichier JSON
# with open('data.json') as file:
#     intents = json.load(file)
with open('data.json', encoding='utf-8') as file:
    intents = json.load(file)


words = []
classes = []
documents = []
ignore_words = ['?', '!',  '.']

# Traiter les données
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniser chaque mot
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Ajouter les documents au corpus
        documents.append((w, intent['tag']))
        # Ajouter les classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizer et normaliser chaque mot et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Trier les classes
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Sauvegarder les données prétraitées avec joblib
joblib.dump(words, 'texts.joblib')
joblib.dump(classes, 'labels.joblib')

# Créer les données d'entraînement
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mélanger les données et les convertir en np.array
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)
print("Training data created")

# Créer le modèle
model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiler le modèle
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entraîner le modèle et le sauvegarder
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.keras')  # Sauvegarder en utilisant le format natif Keras

print("model created")
