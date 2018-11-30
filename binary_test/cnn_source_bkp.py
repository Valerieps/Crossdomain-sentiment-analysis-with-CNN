
'''

    
'''

#==================== LOAD DATASET / TOKENIZATION ==============================
x_all, y_all = [], []
with open('../data/apparel/5-shuffled_class_data.out', 'r') as file:
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

#==================== BINARIZATION ==============================

y_all_binary=[]
for item in y_all:
    if item=='1' or item=='2' or item=='3':
        y_all_binary.append(0)
    if item=='4' or item=='5':
        y_all_binary.append(1)

# print(y_all[:10])
# print(y_all_binary[:10])
# print(len(y_all))
# print(len(y_all_binary))
#
# print(max(y_all_binary), 'maximo')
# print(min(y_all_binary), 'minimo')

# stop


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




#==================== TOP 10.000 FREQUENT WORDS ==============================
from collections import Counter

counter = Counter()

for item in x_all:
    counter.update(item)

counter = counter.most_common(10000)
# print(counter)
# print(type(counter))

vocab, freqs = zip(*counter)
vocab = list(vocab)
vocab.sort()

# print(vocab)

stop


# blz, agora sei quais são as 10k mais freq.
# agora, preciso pegar x_all e trocar todas as que não aparecerem em counter



#==================== CRIAR UM VOCABULARIO ==============================

# para cada sentença
#     para cada palavra da sentenca
#         jogar a palavra num set

vocab = set()
for sentence in x_all:
    for word in sentence:
        vocab.add(word)

# print(vocab)
print(len(vocab))


vocab = list(vocab)
vocab.sort()
# print(vocab)
#==================== CRIAR WORD INDEX ==============================

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

y_train = y_all_binary[:prop_train]
y_test  = y_all_binary[prop_train+1:]

# print('x train', len(x_train))
# print('x teste', len(x_test))
# print('y train', len(y_train))
# print('y teste', len(y_test))
# print(y_train[0])



#==================== MAKE NUMPY ARRAYS ==============================
import numpy as np

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

# print(type(x_train))
# print(len(vocab))
# print(y_train[0])




#==================== MODELO ==============================


from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, Dropout, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard

epochs = 5
batch = 64


# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(14146, embedding_vecor_length, input_length=maior))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(MaxPooling1D(pool_size=2))
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

model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[tensorBoardCallback], batch_size=batch, validation_split=0.1)
model.save_weights("source_apparel.h5")

# Evaluation on the test set
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))