import sys
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential, InputLayer, Model
from keras.layers import LSTM, Embedding
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Convolution1D, SpatialDropout1D, RepeatVector, Activation
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras import backend as K


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
        token_sequences.append(token_sequence)
    return token_sequences

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
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
for line in open("de-dlexdb.data.txt"):
    seg_word = line.strip().split(u"\t")[1]

    encoded_word = []
    annotation = []
    for i in range(1, len(seg_word)):
        prev_char = seg_word[i-1]
        curr_char = seg_word[i]
        if prev_char not in l_i:
            if prev_char not in c_i:
                c_i[prev_char] = index_c
                c_i[index_c] = prev_char
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
        c_i[index_c] = prev_char
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
# model
#inputs = Input(shape=(len(c_i),))
#embedd = Embedding(len(c_i),32)(inputs)
#outputs = TimeDistributed(Dense(len(l_i), activation='softmax'))(embedd)
#model = Model(inputs,outputs)


model = Sequential()
model.add(InputLayer(input_shape=(max_i-1, )))
model.add(Embedding(len(c_i)+1, 16))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(TimeDistributed(Dense(len(l_i))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
             metrics=['accuracy', ignore_class_accuracy(5)])

print(model.summary())

x_train_padded = pad_sequences(x_train,value=len(c_i)-1,maxlen=max_i-1)
y_train_padded = pad_sequences(y_train,value=len(l_i)-1,maxlen=max_i-1)
y_train_padded_cat = to_categorical(y_train_padded,len(l_i))

x_test_padded = pad_sequences(x_test,value=len(c_i)-1,maxlen=max_i-1)
y_test_padded = pad_sequences(y_test,value=len(l_i)-1,maxlen=max_i-1)
y_test_padded_cat = to_categorical(y_test_padded,len(l_i))

x_val_padded = pad_sequences(x_test[20:30],value=len(c_i)-1,maxlen=max_i-1)
y_val_padded = pad_sequences(y_test[20:30],value=len(l_i)-1,maxlen=max_i-1)
y_val_padded_cat = to_categorical(y_val_padded,len(l_i))

model.fit(x_train_padded,y_train_padded_cat,batch_size=50,epochs=20,shuffle=True,validation_data=(x_test_padded, y_test_padded_cat))


testword = "".join(i_c[x] for x in x_test[10])
print(testword)

pred = model.predict(x_val_padded)
print(pred.shape)
pred = logits_to_tokens(pred,i_l)
for i in range(0,len(pred)):
    gt = "".join(i_c[x] for x in encoded_words[20+i])
    pr = "".join(x for x in pred[i][-len(gt):])
    print("%s\n%s" % (gt,pr))

sys.exit(0)

for n_batch, encoded_word in bar(enumerate(x_test)):
    annotation = y_test[n_batch]
    # View each sentence as a batch
    encoded_word = np.array([encoded_word])
    
    pred = model.predict_on_batch(encoded_word)
    pred = np.argmax(pred,-1)[0]
    labels_pred_val.append(pred)
    if pred.tolist() == annotation:
        out = "".join(i_c[x] for x in encoded_word[0].tolist())
        correct += 1
print(correct*100/len(x_test))
