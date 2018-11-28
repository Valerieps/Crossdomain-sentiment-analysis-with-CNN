import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)


word_index = reuters.get_word_index()
# dim = 20000
dim = 0
for i, sentence in enumerate(x_train):
    if len(sentence)>dim:
        dim = len(sentence)

print(dim)


# stop
num_classes = max(y_train)+1
tokenizer = Tokenizer(num_words=dim)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# tam_train = len(x_train[0])
# tam_test = len(x_test[0])

#========TREAT INPUT=======

# print(num_words)

#========INSPECTION=======
# print(x_train[0][0])
# print(y_train[0])

print(keras.backend.shape(x_train))
print(len(x_train))
print(len(x_train[0]))

#========MODEL=======
batch_size = 128
epochs = 1

#input shape é o tamanho do vetor de entrada, 

model = Sequential()

# 512 é a qtd de neurons na 1 camada oculta
# a primeira camada sempre passa o shape ou dim, que são a camada de entrada da rede SEMPRE
model.add(Embedding(dim, output_dim=100 )) #input_dim=dim
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose = 1)

print(f'Test loss:{score[0]}')
print(f'Test accuracy:{score[1]}')