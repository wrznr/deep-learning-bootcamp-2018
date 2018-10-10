import sys
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential, InputLayer, Model
from keras.layers import LSTM, Embedding
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Convolution1D, SpatialDropout1D, MaxPooling1D, Activation
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
        token_sequences.append(token_sequence)
    return token_sequences

def ignore_class_accuracy(to_ignore=0,to_ignore2=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(~(K.equal(y_pred_class, to_ignore) | K.equal(y_pred_class, to_ignore2)), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy


import progressbar

l_i = { u'_' : 0, u'#' : 1, u'+' : 2, u'~' : 3, u'<' : 4, u'!' : 5}
i_l = { 0 : u'_', 1 : u'#', 2 : u'+', 3 : u'~', 4 : u'<', 5 : u'!'}

c_i = {}
index_c = 0
encoded_words = []
annotations = []
max_i = 0
for line in open("de-dlexdb.data_corrected.txt"):
    seg_word = line.strip().split(u"\t")[1]

    encoded_word = []
    annotation = []
    for i in range(1, len(seg_word)):
        prev_char = seg_word[i-1]
        curr_char = seg_word[i]
        if prev_char not in l_i:
            if prev_char not in c_i:
                c_i[prev_char] = index_c
                index_c += 1
            encoded_word.append(c_i[prev_char])
            if curr_char in l_i:
                annotation.append(l_i[curr_char])
            else:
                annotation.append(0)
        if max_i < i:
            max_i = i

    prev_char = seg_word[-1]
    if prev_char not in c_i:
        c_i[prev_char] = index_c
        index_c += 1

    encoded_word.append(c_i[prev_char])
    annotation.append(0)
    assert(len(encoded_word) == len(annotation))
    encoded_words.append(encoded_word)
    annotations.append(annotation)

i_c = {c_i[k]:k for k in c_i}

#
# split
x_train, x_test, y_train, y_test = train_test_split(encoded_words, annotations, test_size=0.1, random_state=2018)

print(len(x_train))
print(len(i_c))
words_val = [list(map(lambda x: i_c[x], w)) for w in x_test]
labels_val = [list(map(lambda x: i_l[x], y)) for y in y_test]

#
# model 1: padding

reg_weight = 1.e-4

model = Sequential()
model.add(Embedding(len(c_i)+1, 8))
model.add(Convolution1D(16,5,padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_weight)))
model.add(Convolution1D(32,5,padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_weight)))
model.add(Convolution1D(64,5,padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_weight)))
model.add(Convolution1D(128,5,padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_weight)))
model.add(Dropout(0.25))
#model.add(GRU(32, return_sequences=True))
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(l_i), activation='softmax')))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
             metrics=['accuracy', ignore_class_accuracy(0,5)])

print(model.summary())

x_train_padded = pad_sequences(x_train,value=len(c_i),maxlen=max_i-1,padding='post')
y_train_padded = pad_sequences(y_train,value=len(l_i)-1,maxlen=max_i-1,padding='post')
y_train_padded_cat = []
for padded_seq in y_train_padded:
    y_train_padded_cat.append(np_utils.to_categorical(padded_seq,len(l_i)))
y_train_padded_cat = np.array(y_train_padded_cat)

x_test_padded = pad_sequences(x_test,value=len(c_i),maxlen=max_i-1,padding='post')
y_test_padded = pad_sequences(y_test,value=len(l_i)-1,maxlen=max_i-1,padding='post')

y_test_padded_cat = []
for padded_seq in y_test_padded:
    y_test_padded_cat.append(np_utils.to_categorical(padded_seq,len(l_i)))
y_test_padded_cat = np.array(y_test_padded_cat)

history = model.fit(x_train_padded,y_train_padded_cat,batch_size=128,epochs=15,shuffle=True,validation_data=(x_test_padded, y_test_padded_cat))

p = model.predict(x_test_padded)
p = np.argmax(p, axis=-1)
eq = np.argmin(p == y_test_padded, axis=-1)
print(np.sum(np.where(eq>0, 1, 0))/y_test_padded.shape[0])

sys.exit(0)


#correct = 0
#bar = progressbar.ProgressBar(maxval=len(x_test_padded))
#for i,_ in bar(enumerate(x_test_padded)):
#    annotation = "".join(i_l[x] for x in y_test[i])
#    pred = model.predict_on_batch(x_test_padded[i:i+1])
#    pred = logits_to_tokens(pred,i_l)
#
#    gt = "".join(i_c[x] for x in encoded_words[i])
#    pr = "".join(x for x in pred[0][0:len(gt)])
#    if pr == annotation:
#        correct += 1
#print(correct*100/len(x_test))

#
# model 2: train on batch

model2 = Sequential()
model2.add(Embedding(len(c_i)+1, 32))
model2.add(Convolution1D(16,5,padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(GRU(32, return_sequences=True))
#model2.add(Bidirectional(GRU(64, return_sequences=True)))
model2.add(TimeDistributed(Dense(len(l_i), activation='softmax')))
 
model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
             metrics=['accuracy', ignore_class_accuracy(5)])

print(model2.summary())

n_epochs = 1
for i in range(n_epochs):
    print("Training epoch {}".format(i))
    
    bar = progressbar.ProgressBar(maxval=len(x_train))
    for n_batch, encoded_word in bar(enumerate(x_train)):
        annotation = np.array(y_train[n_batch])
        # Make annotation one hot
        annotation = np_utils.to_categorical([annotation],len(l_i))
        # View each sentence as a batch
        encoded_word = np.array(encoded_word)[np.newaxis,:]
        model2.train_on_batch(encoded_word, annotation)

    bar = progressbar.ProgressBar(maxval=len(x_test))
    correct = 0
    for n_batch, encoded_word in bar(enumerate(x_test)):
        annotation = y_test[n_batch]
        # View each word as a batch
        encoded_word = np.array(encoded_word)[np.newaxis,:]
        
        pred = model2.predict_on_batch(encoded_word)
        pred = np.argmax(pred,-1)[0]
        if pred.tolist() == annotation:
            correct += 1
    print(correct*100/len(x_test))
