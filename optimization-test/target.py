from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, Dropout, MaxPooling1D
from keras.layers.embeddings import Embedding

'''
    CNN for sentiment analysis on Amazon multi-domain dataset
    Designed and normalized to work with cross-domain SA
    by vesperM

'''
most_freq_words = 7489
pesos = 'source_sports_7489.h5'
input_file = '../data/grocery/5-shuffled_class_data.out'

# ==================== LOAD DATASET / TOKENIZATION ==============================
print('Loading dataset...')
x_all, y_all = [], []
with open(input_file, 'r') as file:
    for line in file:
        label = line[0]
        y_all.append(label)

        data = line[2:].strip()
        words = data.split(' ')
        x_all.append(words)

# print(y_all[0])
# print(type(x_all)) #x_all é uma lista de listas-de-palavras
# print(len(x_all))
# print(len(x_all[0]))


# ==================== BINARIZATION ==============================
print('Binarization...')
y_all_binary = []
for item in y_all:
    if item == '1' or item == '2' or item == '3':
        y_all_binary.append(0)
    if item == '4' or item == '5':
        y_all_binary.append(1)

# print(y_all[:10])
# print(y_all_binary[:10])
# print(len(y_all))
# print(len(y_all_binary))
#
# print(max(y_all_binary), 'maximo')
# print(min(y_all_binary), 'minimo')

# stop


# ==================== LEMMATIZATION ==============================
print('Lemmatization...')
lemmatizer = WordNetLemmatizer()

# print(x_all[0])
for i, sentence in enumerate(x_all):
    for j, word in enumerate(sentence):
        word2 = lemmatizer.lemmatize(word)
        x_all[i][j] = word2
# print(x_all[0])


# ==================== PADDING ==============================
print('Padding...')

maior = 400

for sentence in x_all:
    if len(sentence) < maior:
        tam = maior - len(sentence)
        for i in range(tam):
            sentence.append('PAD')
    elif len(sentence) > maior:
        tam = len(sentence) - maior
        for i in range(tam):
            del sentence[-1]

confere = True
for sentence in x_all:
    if len(sentence)==maior:
        confere = confere and True
    else:
        confere = confere and False
print(confere)


# ==================== TOP 10.000 FREQUENT WORDS ==============================
print('Selecting 10k most frequent words...')
counter = Counter()

for item in x_all:
    counter.update(item)

counter = counter.most_common(most_freq_words)
# print(counter)
# print(type(counter))

vocab, freqs = zip(*counter)
vocab = list(vocab)
vocab.append('OOV')
vocab.sort()
tam_vocab = len(vocab)

print(tam_vocab)


# ==================== SUBSTITUIR OOV ==============================
print('Replacing OOV...')

for i, sentence in enumerate(x_all):
    for j, word in enumerate(sentence):
        if word not in vocab:
            x_all[i][j] = 'OOV'

# print(x_all[0])
# stop


# ==================== CRIAR WORD INDEX ==============================
print('Coding words...')

word2index = {}

for i, item in enumerate(vocab):
    word2index[item] = i
# print(word2index)

# ==================== CODIFICAR PALAVRAS ==============================
x_all_coded = []
for sentence in x_all:
    lista_coded = []
    for word in sentence:
        # buscar o indice da palavra e substituir
        # fazer append numa outra lista
        aux = word2index[word]
        lista_coded.append(aux)
    x_all_coded.append(lista_coded)

# ==================== DIVIDIR EM TRAIN TEST ==============================

prop_train = int(len(x_all_coded) * 0.8)

x_train = x_all_coded[:prop_train]
x_test = x_all_coded[prop_train + 1:]

y_train = y_all_binary[:prop_train]
y_test = y_all_binary[prop_train + 1:]

# print('x train', len(x_train))
# print('x teste', len(x_test))
# print('y train', len(y_train))
# print('y teste', len(y_test))
# print(y_train[0])


# ==================== MAKE NUMPY ARRAYS ==============================

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

# print(type(x_train))
# print(len(vocab))
# print(y_train[0])


# ==================== MODELO ==============================

epochs = 6
batch = 64

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(tam_vocab, embedding_vecor_length, input_length=maior))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.load_weights(pesos)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluation on the test set
scores = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))
