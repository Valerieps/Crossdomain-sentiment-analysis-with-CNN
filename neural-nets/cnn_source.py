# CNN for sequence classification in the IMDB dataset
import nltk
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard


'''
    OK 1. load dataset dividido em class e data
    OK 1. Tokenizar x_all
    OK 2. Fazer padding
    OK 3. Transformar o vetor data: palavras representadas por índices
        - criar dicionário de word_index (deixar o 0 para PAD)
    
    OK 4. Dividir em train, test
    5. Transformar tudo pra numpy array
'''

#==================== LOAD DATASET ==============================
x_all, y_all = [], []
with open('../data/apparel_treated_class_data.out', 'r') as file:
    for line in file:
        label = line[0]
        y_all.append(label)

        data = line[2:].strip()
        words = data.split(' ')
        x_all.append(words)

# print(y_all[:5])
# print(type(x_all)) #x_all é uma lista de listas-de-palavras
# print(len(x_all))
# print(len(x_all[0]))

#==================== LEMMATIZATION ==============================
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# print(x_all[0])
for i, sentence in enumerate(x_all):
    for j, word in enumerate(sentence):
        word2 = lemmatizer.lemmatize(word)
        x_all[i][j] = word2
# print(x_all[0])


#==================== PADDING ==============================
# pegar o maior len de x_all

maior = 0
for i, sentence in enumerate(x_all):
    if len(x_all[i]) > maior:
        maior = len(x_all[i])
        qual = i

# print(maior)
# print(qual)
# print(x_all[qual])
# print(len(x_all[qual]))

for sentence in x_all:
    if len(sentence)<maior:
        tam = maior - len(sentence)
        for i in range(tam):
            sentence.append('PAD')

# print(len(x_all[0]))



#==================== CRIAR UM VOCABULARIO ==============================

# para cada sentença
#     para cada palavra da sentenca
#         jogar a palavra num set

vocab = set()
for sentence in x_all:
    for word in sentence:
        vocab.add(word)

# print(vocab)
# print(len(vocab))

vocab = list(vocab)
vocab.sort()
# print(vocab)

word2index={}

for i, item in enumerate(vocab):
    word2index[item] = i
# print(word2index)

#==================== CODIFICAR PALAVRAS ==============================
x_all_coded=[]
for sentence in x_all:
    lista_coded = []
    for word in sentence:
        # buscar o indice da palavra e substituir
        # fazer append numa outra lista
        aux = word2index[word]
        lista_coded.append(aux)
    x_all_coded.append(lista_coded)


#==================== DIVIDIR EM TRAIN TEST ==============================

prop_train = int(len(x_all_coded)*0.8)

x_train = x_all_coded[:prop_train]
x_test  = x_all_coded[prop_train+1:]

y_train = x_all_coded[:prop_train]
y_test  = x_all_coded[prop_train+1:]

# print('x train', len(x_train))
# print('x teste', len(x_test))
# print('y train', len(y_train))
# print('y teste', len(y_test))


#==================== MAKE NUMPY ARRAYS ==============================
import numpy as np

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

print(type(x_train))


stop
#==================== MODELO ==============================



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding


# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

# Evaluation on the test set
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))