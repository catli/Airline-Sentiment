
from process_data import ProcessData, createFTEmbedding
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

import pandas as pd

import pdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def trainPredModel(data):
    '''
        Split dataset to training and evaluation (20% evaluation split)
        Fit model to training data, and generate a doc with evaluation
        as well as the summary of errors
    '''
    process = ProcessData(sentdata)
    padded_sequence, tokenizer, labels = process.createTrainData()

    ft = createFTEmbedding()
    ft.processDict(tokenizer)

    x_train, x_val, y_train, y_val  = train_test_split(padded_sequence, labels)

    model = compileModel(padded_sequence, labels, tokenizer,
        ft.embedding_matrix)

    model.fit(x_train, y_train,
              batch_size=20,
              epochs=10,
              validation_data=(x_val, y_val))

    y_pred = model.predict(x_val)
    translateEvalData(y_pred, y_val, x_val, tokenizer)
    return model, x_train, x_val, y_train, y_val, y_pred



def compileModel(padded_sequence, labels, tokenizer, embedding_matrix):
    '''
        compile the model that will be used for training
    '''
    max_sequence_length = len(padded_sequence[0])
    num_words = len(tokenizer.word_index) + 1
    embedding_dim = 300
    embedding_layer = Embedding(input_dim = num_words,
                                output_dim = embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = LSTM(128, dropout=0.2,
        recurrent_dropout=0.2, return_sequences = False)(embedded_sequences)
    x = Dropout(0.5)(x)
    # x = Conv1D(50, 5, activation = 'relu')(embedded_sequences)
    # x = MaxPooling1D(5)(x)
    # x = Dropout(0.5)(x)
    # x = Flatten()(x)
    x = Dense(5, activation='relu')(x)
    preds = Dense( labels.shape[1], activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def translateEvalData(y_pred, y_val, x_val, tokenizer):
    '''
        convert the prediction and actual values to readable form and write
        to file
    '''
    back_index = {}
    for key, val in tokenizer.word_index.items():
        back_index[val] = key
    pred_label = pd.DataFrame(y_pred).apply(lambda x: x.idxmax(), axis=1)
    y_label = pd.DataFrame(y_val).apply(lambda x: x.idxmax(), axis=1)
    is_correct = y_label==pred_label

    tabulatePrediction(y_label, is_correct)

    output_writer = open('eval_prediction.csv', 'w')

    output_writer.write('actual sentiment' + '\t' +
                    'predicted sentiment' + '\t' +
                    'words' + '\t' +
                    'correct' + '\n')

    for pred, y, x, b in zip(pred_label, y_label, x_val, is_correct):
        words = translateArrToWords(x, back_index)
        output_writer.write(str(y) + '\t' +
               str(pred) + '\t' +
               str(words) + '\t' +
               str(b) + '\n' )

def tabulatePrediction(y_label, is_correct):
    '''
        percent correct prediction by actual label
    '''
    prediction = pd.DataFrame({'actual': y_label, 'is_correct': is_correct})
    print('PREDICTION TABULATION')
    print(prediction.groupby(['actual']).mean())


def translateArrToWords(arr, back_index):
    '''
        function to translate tokenized indices back to human-readable words
    '''
    words = ''
    for ind in arr:
        words = words + str((back_index[ind] + ' ') if ind!=0 else '')
    return words


if __name__ == "__main__":
    sentdata = pd.read_csv('data/train.csv')
    model, x_train, x_val, y_train, y_val, y_pred = trainPredModel(sentdata)



