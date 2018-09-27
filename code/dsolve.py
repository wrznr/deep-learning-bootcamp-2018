import sys
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D

import progressbar

l_i = { u'_' : 0, u'#' : 1, u'+' : 2, u'~' : 3, u'<' : 4}
i_l = { 0 : u'_', 1 : u'#', 2 : u'+', 3 : u'~', 4 : u'<'}

c_i = {}
index_c = 0
encoded_words = []
annotations = []
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
model = Sequential()
model.add(Embedding(len(c_i),32))
model.add(Dropout(0.25))
model.add(SimpleRNN(32,return_sequences=True))
model.add(TimeDistributed(Dense(len(l_i), activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')
print(model.summary())

#
# train
n_epochs = 5
for i in range(n_epochs):
    print("Training epoch {}".format(i))
    
    bar = progressbar.ProgressBar(maxval=len(x_train))
    for n_batch, encoded_word in bar(enumerate(x_train)):
        annotation = np.array(y_train[n_batch])
        # Make annotation one hot
        annotation = np.eye(len(l_i))[annotation][np.newaxis,:]
        # View each sentence as a batch
        encoded_word = np.array(encoded_word)[np.newaxis,:]
        
        model.train_on_batch(encoded_word, annotation)

    labels_pred_val = []
    bar = progressbar.ProgressBar(maxval=len(x_test))
    correct = 0
    for n_batch, encoded_word in bar(enumerate(x_test)):
        annotation = y_test[n_batch]
        # View each sentence as a batch
        encoded_word = np.array(encoded_word)[np.newaxis,:]
        
        pred = model.predict_on_batch(encoded_word)
        pred = np.argmax(pred,-1)[0]
        labels_pred_val.append(pred)
        if pred.tolist() == annotation:
            correct += 1
    print(correct*100/len(x_test))



    labels_pred_val = [list(map(lambda x: i_l[x], y)) for y in labels_pred_val]
    #con_dict = conlleval(labels_pred_val, annotations, words_val, 'measure.txt')
    #print('Precision = {}, Recall = {}, F1 = {}'.format(con_dict['r'], con_dict['p'], con_dict['f1']))
